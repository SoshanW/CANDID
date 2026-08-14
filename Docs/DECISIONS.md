# Decision Log

**Project:** Condition-Dependent Proxy-Label Noise and Calibrated Abstention for Trustworthy Multi-Condition Mental-Health Screening
**Module:** 6COSC023W, Informatics Institute of Technology (University of Westminster)
**Author:** Soshan
**Supervisor:** Sachindu Jayasinghe
**Repository:** github.com/SoshanW/mental-health-screening-fyp
**Log started:** 2026-07-17 (retrospective; earlier entries reconstructed)

---

## Purpose

This file records **what was decided, why, on what evidence, and what happened when a decision turned out to be wrong.** It follows the Architecture Decision Record (ADR) convention.

It exists for three reasons:

1. **Viva defence.** When an examiner asks "why did you not just estimate the transition matrix from the data?", the answer should be a dated entry with evidence, not a recollection.
2. **Demonstrating rigour.** The reversal entries (D-020 through D-026) are the most valuable part of this file. A project that never changed its mind is a project that never checked.
3. **Not relitigating.** Several directions were considered and closed. They stay closed unless new evidence appears.

### Status legend

| Status | Meaning |
|---|---|
| **ACTIVE** | Current, load-bearing. |
| **SUPERSEDED** | Replaced by a later decision. Kept for the record. |
| **REVERSED** | Found to be wrong. The reversal reasoning is the point. |
| **OPEN** | Not yet decided. Needs action. |
| **CLOSED** | Considered and rejected. Do not reopen without new evidence. |

### Template for new entries

```markdown
## D-0NN · [Title]
**Date:** YYYY-MM-DD · **Status:** ACTIVE · **Category:** [Scope | Data | C1 | C2 | C3 | Engineering | Writing]

**Decision.** One sentence, in the past tense.

**Context.** What prompted this. What was true at the time.

**Reasoning.** Why this rather than the alternatives.

**Evidence.** Citations, measurements, or "none, this is a judgement call."

**Consequences.** What this commits the project to. What it forecloses.

**Links.** Supersedes / superseded by / depends on.
```

### Date honesty

Entries D-001 to D-019 are **reconstructed retrospectively on 2026-07-17** and are marked *undated*. Backfill real dates from the repository git log and supervisor meeting notes before submission. An examiner will not mind reconstructed dates; they will mind invented ones.

---

## Index

| ID | Decision | Status | Category |
|---|---|---|---|
| D-001 | Frame as screening and decision-support, not diagnosis | ACTIVE | Scope |
| D-002 | Three coupled contributions rather than one system | ACTIVE | Scope |
| D-003 | Treat class imbalance as evidence, not an engineering nuisance | ACTIVE | Scope |
| D-004 | Low et al. as the sole Reddit source | ACTIVE | Data |
| D-005 | Exclude SWMH | ACTIVE | Data |
| D-006 | SMHD and RSDD unavailable, do not pursue | CLOSED | Data |
| D-007 | DAIC-WoZ as the depression-only clinical anchor | ACTIVE | Data |
| D-008 | Reject E-DAIC | CLOSED | Data |
| D-009 | Aich et al. as a stretch goal, not a dependency | OPEN | Data |
| D-010 | Author-grouped split, not stratified | ACTIVE | Engineering |
| D-011 | MentalBERT with class-weighted loss as the baseline | ACTIVE | Engineering |
| D-012 | `src/` package in VS Code; notebooks as thin GPU runners | ACTIVE | Engineering |
| D-013 | Follow the "Jazzify" thesis structure | ACTIVE | Writing |
| D-014 | Adopt supervisor's prose rules | ACTIVE | Writing |
| D-015 | Abandon XAI, distillation, GNN and LLM-judge directions | CLOSED | Scope |
| D-016 | Drop "CALM" as a project name | CLOSED | Writing |
| D-017 | Verify novelty by search, never from assumption | ACTIVE | Scope |
| D-018 | Two-regime identifiability framing | ACTIVE | C1 |
| D-019 | Name scope and limitations before examiners raise them | ACTIVE | Scope |
| D-020 | **REVERSED:** "nobody couples noise estimation to abstention" | REVERSED | C2 |
| D-021 | **REVERSED:** the invented `g()` coupling function | REVERSED | C2 |
| D-022 | **REVERSED:** posterior width as the coupling signal | REVERSED | C2 |
| D-023 | **REVERSED:** Dirichlet prior stabilises the conformal quantile | REVERSED | C2/C3 |
| D-024 | **REVERSED:** "every route to the matrix needs clean labels" | REVERSED | C2 |
| D-025 | **REVERSED:** the α_V = 0 proposition with pruning | REVERSED | C2 |
| D-026 | **REVERSED:** PHQ-8 described as clinician-administered | REVERSED | Data |
| D-027 | Build on Sesia et al. rather than compete with it | ACTIVE | C2 |
| D-028 | Elicit in Penso's direction, convert to Sesia's | ACTIVE | C2 |
| D-029 | Keep Mondrian (label-conditional), not marginal coverage | ACTIVE | C2 |
| D-030 | C1 becomes a diagnostic, not an estimator | ACTIVE | C1 |
| D-031 | Do not prune the elicited set (preserve α_V = 0) | OPEN | C2 |
| D-032 | Three-arm experiment as the core evaluation | ACTIVE | C2 |
| D-033 | Read primary sources before any claim becomes load-bearing | ACTIVE | Scope |
| D-034 | Clusterability diagnostic results: prediction falsified | ACTIVE | C1 |
| D-035 | Base-embedding control run to quantify the fine-tuning artifact | ACTIVE | C1 |
| D-036 | Base-embedding control result and Stage 2 pre-registration | ACTIVE | C1 |
| D-037 | HOC full result: prediction held, estimator internally healthy | ACTIVE | C1 |
| D-038 | Cleanlab cross-validation fold integrity verified | ACTIVE | C1 |
| D-039 | Cleanlab result: agrees with HOC on structure, contradicts on magnitude | ACTIVE | C1 |
| D-040 | HOC on representations not trained on the project's labels (pre-registration) | ACTIVE | C1 |
| D-041 | D-040 result: HOC's structure is stable, its magnitude is representation-dependent | ACTIVE | C1 |
---

# Part 1 · Scope and framing

## D-001 · Frame as screening and decision-support, not diagnosis
**Date:** undated · **Status:** ACTIVE · **Category:** Scope

**Decision.** The system is described throughout as screening and decision-support research. It is never described as a diagnostic tool.

**Reasoning.** The labels are proxy labels (subreddit membership), not clinical diagnoses. A system trained on proxy labels cannot make diagnostic claims without a category error. Institutional ethics approval is required before data handling, and the framing must be consistent with what was approved.

**Evidence.** Chancellor and De Choudhury (2020, *npj Digital Medicine* 3:43) document construct-validity failures across the field: of 75 reviewed studies, only 32 (42%) reported enough to be reproducible.

**Consequences.** Every deliverable must use this language consistently. It also constrains the abstention framing: abstention is deferral to a clinician, not a refusal to diagnose.

---

## D-002 · Three coupled contributions rather than one system
**Date:** undated · **Status:** ACTIVE · **Category:** Scope

**Decision.** The project makes three contributions (C1 noise model, C2 calibrated abstention, C3 limits analysis) that depend on each other, rather than building one large system.

**Reasoning.** The marking rubric weights Design at 0.5 and Introduction at 0.4, rewarding methodological contribution over system scale. A coupled trio is defensible as a single intellectual move; three separate gadgets would not be.

**Consequences.** The coupling is the thing being defended. If the coupling fails empirically, the project falls back on C3 (see D-032).

---

## D-003 · Treat class imbalance as evidence, not an engineering nuisance
**Date:** undated · **Status:** ACTIVE · **Category:** Scope

**Decision.** The roughly 20:1 depression-to-bipolar imbalance is treated as supporting evidence for the thesis that condition-dependent noise priors are necessary, not merely as a problem to engineer around.

**Reasoning.** The conditions with the least data are the conditions whose noise is hardest to estimate. That is not a coincidence, it is the argument.

**Consequences.** Strengthened considerably by later findings (see D-030 and the prevalence arithmetic in Part 4). The imbalance turned out to be the reason the competing estimators have no guarantee on three of four conditions.

---

## D-015 · Abandon XAI, distillation, GNN and LLM-judge directions
**Date:** undated · **Status:** CLOSED · **Category:** Scope

**Decision.** The following were considered and closed: pure XAI/conformal-prediction faithfulness evaluation; MetroXAI / measurement-error-aware XAI; calibrated knowledge distillation; conformal GNNs; hallucination probes; LLM-judge debiasing.

**Consequences.** Do not resurrect without an explicit reason. Each was closed for its own reasons and reopening one costs time that the current direction needs.

---

## D-016 · Drop "CALM" as a project name
**Date:** undated · **Status:** CLOSED · **Category:** Writing

**Decision.** "CALM" was rejected as a project name; it is crowded in the broader ML literature. "NoCAB" was flagged as a more distinctive alternative but has not been committed to.

**Status note.** Naming remains uncommitted. Low priority.

---

## D-017 · Verify novelty by search, never from assumption
**Date:** undated · **Status:** ACTIVE · **Category:** Scope

**Decision.** No claim that a gap is open may be made without an actual literature search confirming it. Re-verify before final submission.

**Reasoning.** Established after Claude repeatedly overclaimed open gaps from training knowledge rather than evidence.

**Consequences.** This decision is the reason D-020 and D-024 were caught rather than surviving to the viva. **It has been the single most valuable process decision in the project.** See Part 4 for what it caught.

**Amendment (2026-07-17).** Search is necessary but not sufficient. See D-033: search results and abstracts also proved unreliable. The rule is now *read the primary source*.

---

## D-019 · Name scope and limitations before examiners raise them
**Date:** undated · **Status:** ACTIVE · **Category:** Scope

**Decision.** Every known weakness is stated in the text before an examiner can find it.

**Reasoning.** Stated preference for honest framing over overclaiming. Also strategic: a named limitation is a demonstration of understanding; a discovered one is a failure of it.

**Current instances.** The elicitation-validity assumption (D-031); the Ding pooling tension; the Theorem 1 versus Theorem 2 distinction in the prevalence table; the DAIC-WoZ construct-validity problems.

---

## D-033 · Read primary sources before any claim becomes load-bearing
**Date:** 2026-07-17 · **Status:** ACTIVE · **Category:** Scope

**Decision.** No claim about a paper's content enters the thesis based on its abstract, its citation in another paper, or a search-result snippet. The PDF gets read.

**Context.** Over the course of one working session, four separate claims about the literature were made from abstracts and snippets. **All four were materially wrong** (D-020, D-023, D-024, plus the Penso mischaracterisation). Every claim made after reading the full paper has held.

**Evidence.** The reversal record in Part 4 of this file is the evidence.

**Consequences.** Slower. Also the reason the project's central claim is currently defensible rather than fatally wrong.

---

# Part 2 · Data

## D-004 · Low et al. as the sole Reddit source
**Date:** undated · **Status:** ACTIVE · **Category:** Data

**Decision.** The Low et al. Reddit Mental Health Dataset (Zenodo 3941387, PDDL public domain) is the only Reddit source.

**Composition (independently verified).** depression 117,331; eating disorder (EDAnonymous) 14,577; schizophrenia 8,712; bipolar (bipolarreddit) 5,780.

**Reasoning.** It covers all four target conditions at larger scale than the alternatives, under a permissive licence.

---

## D-005 · Exclude SWMH
**Date:** undated · **Status:** ACTIVE · **Category:** Data

**Decision.** SWMH (Ji et al.) is excluded definitionally, not for convenience.

**Reasoning.** Low et al. covers the same conditions at larger scale. Including both would add no coverage and would complicate the harmonisation table.

**Consequences.** Ji et al. (2022, *Neural Computing and Applications* 34(13), 10309–10319) stays in the literature survey but is **repositioned as prior-art context** rather than as a dataset source: a multi-condition social-media detection model that performs detection without calibration or abstention.

---

## D-006 · SMHD and RSDD unavailable
**Date:** undated (confirmed June 2026) · **Status:** CLOSED · **Category:** Data

**Decision.** Neither dataset is obtainable. The Reddit API terms-of-service change closed the Georgetown access form.

**Consequences.** Do not suggest, do not pursue, do not cite as a "future work" option without noting unavailability.

---

## D-007 · DAIC-WoZ as the depression-only clinical anchor
**Date:** undated (access confirmed and integrated) · **Status:** ACTIVE · **Category:** Data

**Decision.** DAIC-WoZ (Gratch et al., 2014) is the clinical anchor. 188 participants, PHQ-8 labels, depression only.

**Reasoning.** It is the only obtainable clinical corpus for any of the four conditions.

**Known problems, stated up front.** Construct validity is contested from two independent directions: Burdisso et al. (2024, *Findings of ACL*) show many models exploit the interviewer's prompts rather than participant language; Patapati et al. (2025, ICMI Companion, doi:10.1145/3747327.3763034) argue the single self-reported scale means models may detect general distress rather than depression specifically.

**Consequences.** DAIC-WoZ is treated as **a shift target with its own contamination**, not as a gold standard. This framing is a contribution, not a hedge.

---

## D-008 · Reject E-DAIC
**Date:** undated · **Status:** CLOSED · **Category:** Data

**Decision.** E-DAIC was considered and rejected: marginal benefit, not worth a second gated application.

---

## D-009 · Aich et al. as a stretch goal, not a dependency
**Date:** undated · **Status:** OPEN · **Category:** Data

**Decision.** Aich et al. (2025, CLPsych; 644 participants with bipolar/schizophrenia/healthy-control expert diagnoses) is pursued as an aspirational second clinical anchor. In active negotiation with Natalie Parde (parde@uic.edu, UIC) and the UCSD team.

**Reasoning.** If granted, C1's noise estimates for bipolar and schizophrenia could be validated against expert diagnoses rather than remaining assumptions. That would be transformative. But it cannot be a dependency, because access is outside the project's control.

**Status.** Pending. **The project must be complete and defensible without it.**

**Open item.** The specific LLM model names and metric definitions in Aich et al. remain unverified. Check the ACL Anthology PDF (2025.clpsych-1.15) before citing specifics.

---

## D-026 · REVERSED: PHQ-8 described as clinician-administered
**Date:** 2026-07-17 · **Status:** REVERSED · **Category:** Data

**What was claimed.** Project documentation described DAIC-WoZ's PHQ-8 as "a clinician-administered instrument," contrasted against Reddit's self-selected forum membership.

**What is true.** PHQ-8 in DAIC-WoZ is a **self-report questionnaire completed by the participant**, administered within a structured interview protocol.

**Why it mattered.** The contrast being drawn is real but weaker than claimed: self-selected forum membership versus *a validated self-report instrument under a structured protocol*, not versus a clinician's diagnosis. Overstating the anchor's authority runs directly against D-007's framing and against Patapati et al.'s critique, which the project already cites.

**Fixed in.** Technical contribution explainer v2 and v3.

---

# Part 3 · Engineering

## D-010 · Author-grouped split, not stratified
**Date:** undated · **Status:** ACTIVE · **Category:** Engineering

**Decision.** 80/10/10 split grouped by author using scikit-learn's `GroupShuffleSplit` keyed on `author_id`. Not stratified by condition.

**Reasoning.** If the same author's posts appeared in both train and test, the model could learn to recognise that person's writing style rather than the condition, inflating the score.

**What was verified, and what was not.** Author-grouping prevents leakage but does **not** by construction guarantee equal per-condition proportions across piles. This was checked empirically on the actual splits: per-condition proportions held within **0.6 percentage points** across train, validation and test, with roughly **490 bipolar examples in each of validation and test**.

**Honesty note.** The property holds empirically for this run, not by construction. Re-check whenever the data or split seed changes. **Do not claim stratification the code does not perform.**

**Consequences.** The figure of ~490 has become load-bearing. It is the per-condition conformal calibration sample size, and it determines whether C2 produces useful prediction sets or vacuous ones.

---

## D-011 · MentalBERT with class-weighted loss as the baseline
**Date:** undated (Milestone 0 complete) · **Status:** ACTIVE · **Category:** Engineering

**Decision.** Class-weighted MentalBERT fine-tune via a `WeightedLossTrainer` (HuggingFace `Trainer` subclass), inverse-frequency weighting. 3 epochs, batch size 16, 256-token cap.

**Result.** 96% accuracy, 0.8782 macro-F1 on the held-out test set.

**Reasoning.** Plain cross-entropy under 20:1 imbalance would let the model ignore bipolar entirely and still look accurate.

**Consequences.** The current loss handles *imbalance* but knows nothing about *label noise*.

---

## D-012 · `src/` package in VS Code; notebooks as thin GPU runners
**Date:** undated · **Status:** ACTIVE · **Category:** Engineering

**Decision.** Structured, testable, version-controlled code lives in the `src/` package. Colab and Kaggle clone the repo and call package functions. Notebooks are not the primary code home.

**Reasoning.** The rubric weights software engineering at 50%. Notebooks do not demonstrate it.

**Conventions.** Injectable frozen dataclass configs; pure-logic modules avoiding torch imports where possible; US spelling in code (distinct from UK English in thesis prose); no em dashes; `# DECISION:` comments for key assumptions; full type hints; pytest structure mirroring each module; a single `(source, raw_label) → Condition` harmonisation table with no branching on raw strings elsewhere; `validate_schema()` at the end of every `load()`; `include_interviewer: bool = False` on the DAIC loader for ablation.

---

# Part 4 · The novelty claim and its reversals

> **This is the most important part of this file.** The project's central claim was wrong twice and was corrected twice, both times by reading a primary source. An examiner who reads only this section will conclude the work was done carefully. Do not sanitise it.

## D-020 · REVERSED: "nobody couples noise estimation to abstention"
**Date claimed:** undated · **Date reversed:** 2026-07-17 · **Status:** REVERSED · **Category:** C2

**What was claimed.** That C2's coupling of a per-condition noise estimate to a per-condition abstention threshold was unprecedented. Stated in the project proposal and in the technical explainer as: *"No existing system connects a label-noise estimate to an abstention threshold this way."*

**What killed it.** Sesia, M., Wang, Y.X.R. and Tong, X. (2025) 'Adaptive conformal classification with noisy labels', *Journal of the Royal Statistical Society Series B* 87(3), pp. 796–815, doi:10.1093/jrsssb/qkae114.

Their equations (13) to (15) set a per-class conformal threshold as an explicit function of a per-class noise estimate. Their target is label-conditional (Mondrian) coverage. Their contamination model is a general K x K matrix, not uniform noise.

**How it was caught.** Literature search under D-017, then confirmed by reading the full PDF.

**Why it mattered.** The claim was a **gap claim** (nobody has done this). Gap claims are cheap to defend but fatal when false. An examiner familiar with the conformal literature would have ended the discussion there.

**What replaced it.** A **differentiation claim**, then (after D-024) a structural claim. See D-027.

**Lesson recorded.** A named competitor is worth more than an absence. "Sesia et al. correct coverage back to nominal despite noise; I let condition-dependent noise change the policy" is a stronger sentence than "nobody has done this," because it proves the literature was read.

---

## D-021 · REVERSED: the invented `g()` coupling function
**Date reversed:** 2026-07-17 · **Status:** REVERSED · **Category:** C2

**What was claimed.** C2 defined an abstention threshold as `τ_i = g(α, η_i)` with `∂τ_i/∂η_i > 0`: noisier condition, stricter threshold, more abstention.

**Three problems.**

1. **It was not novel** (see D-020).
2. **It pointed the wrong way.** Sesia's Corollary 1 shows that noisy calibration data already makes standard conformal prediction **over-cover**. Noise makes you conservative for free. Sesia's entire contribution is *removing* that excess conservativeness because it produces uselessly large prediction sets. A `g()` that adds more conservativeness pays twice for something already obtained.
3. **It was invented.** "I made up a function" was always the softest point in the defence.

**What replaced it.** The coupling is now emergent rather than designed: a wide elicited region produces wide prediction sets through the borrowed machinery, with no invented function. See D-027.

---

## D-022 · REVERSED: posterior width as the coupling signal
**Date proposed and reversed:** 2026-07-17 (same session) · **Status:** REVERSED · **Category:** C2

**What was proposed.** After D-021, couple the abstention threshold to the **posterior width** of the Dirichlet noise estimate rather than its mean, on the argument that Sesia's Algorithm 2 tightens sets when the confidence region is wider.

**Why it is wrong.** The Dirichlet-multinomial posterior after counts `n` with prior `α` is `Dir(α + n)`. **Posterior width shrinks as `α` grows, with no data at all.** In the prior-dominated regime (bipolar), the posterior concentrates around the prior and the width goes *narrow*. The coupling would read that as confidence and abstain **less** on bipolar. Exactly backwards.

**The deeper problem.** Posterior width cannot distinguish "narrow because the data pinned it down" from "narrow because I asserted it firmly."

**What replaced it.** A **credal envelope**: the set of estimates induced by sweeping over a range of clinically defensible priors. That set is narrow when data dominates and wide when the prior does, which is the behaviour wanted. This led to D-027.

**Why this entry matters.** It was caught within one session, before any code was written. It is a good example of an idea being killed by its own arithmetic.

---

## D-023 · REVERSED: Dirichlet prior stabilises the conformal quantile
**Date reversed:** 2026-07-17 · **Status:** REVERSED · **Category:** C2/C3

**What was claimed.** That C1's Dirichlet prior does double duty: clinical seeding *and* shrinkage that stabilises the Mondrian conformal quantile at low `n`, making it viable at bipolar's ~490 calibration examples. This was proposed as a strong C2 differentiator, citing Ding et al. (2023, NeurIPS) on erratic per-class quantiles.

**Why it is wrong: two different sample sizes.**

| Quantity | Where it lives | Bipolar n |
|---|---|---|
| Transition matrix estimate | **training** split, via confident joint | ~4,800 |
| Conformal quantile | **calibration** split | ~490 |

The prior shrinks the first. It adds no calibration data, so it **cannot touch the second**. Ding's instability hits all experimental arms identically.

**Consequences.** Ding et al. is **not a C2 differentiator**. It was moved to C3, where it belongs, as one of several independent lines predicting breakdown on the same conditions.

**Compounding note.** This claim was made from a search snippet, not the paper. See D-033.

---

## D-024 · REVERSED: "every route to the matrix needs clean labels"
**Date claimed:** 2026-07-17 · **Date reversed:** 2026-07-17 · **Status:** REVERSED · **Category:** C2

**What was claimed.** After D-020, the replacement claim was: Sesia's algorithm needs a noise matrix, and every route to obtaining one requires clean labels, which do not exist for bipolar or schizophrenia. This became the central claim of technical explainer v2.

**What killed it.** Zhu, Z., Song, Y. and Liu, Y. (2021) 'Clusterability as an alternative to anchor points when learning with noisy labels', *Proceedings of the 38th ICML*, PMLR 139, pp. 12912–12923.

Their HOC estimator obtains a **unique** transition matrix from noisy labels alone: no clean data, no anchor points. It uses third-order consensus among 2-nearest-neighbour noisy labels. Code is public at github.com/UCSC-REAL/HOC. Their Theorem 1 proves uniqueness under 2-NN label clusterability, non-singularity and diagonal dominance.

**How it was caught.** The paper was identified as a threat and the PDF was obtained and read.

**Why the reversal improved the project.** The follow-up paper by the same authors supplies the rescue, and it is stronger than the original claim. See D-030.

---

## D-025 · REVERSED: the α_V = 0 proposition with pruning
**Date proposed and reversed:** 2026-07-17 · **Status:** REVERSED (partially) · **Category:** C2

**What was claimed.** A proposition stating that a clinically elicited region, **pruned** against the observed subreddit proportions, satisfies Sesia's Theorem 4 with `α_V = 0`, giving a deterministic containment guarantee.

**The flaw.** The proof asserted that the true model "is by definition consistent with the observed proportions, so it survives the pruning." **This is false.** The observed proportions are estimated from a *finite sample*. The true model is consistent with the *population* proportions, not necessarily with the sample. Sampling noise alone could prune the true model out, after which the region no longer contains the truth and the guarantee collapses.

**Two repairs, both legitimate, and the choice is a real design decision.**

| Option | Containment | α_V | Cost |
|---|---|---|---|
| **(a) Do not prune** | Deterministic | 0 | Wider region, wider prediction sets |
| **(b) Prune against a confidence region for the proportions** | Probabilistic | α_ρ > 0 | Tighter region, but the "tighter than their bootstrap" corollary dies |

**Current position.** Option (a) is stated in the proposition. See D-031, which remains OPEN.

**Note.** With ~140k posts, `α_ρ` can be made tiny. But tiny is not zero, and the entire point of a finite-sample guarantee is not waving that away.

---

# Part 5 · Current position (post-reversal)

## D-027 · Build on Sesia et al. rather than compete with it
**Date:** 2026-07-17 · **Status:** ACTIVE · **Category:** C2

**Decision.** Sesia, Wang and Tong (2025) is the foundation, not the competitor. The contribution fills an input slot their own Theorem 4 leaves open.

**The argument.** Theorem 4 requires a region `[V_low, V_upp]` containing the true noise parameters with probability at least `1 - α_V`, independent of the calibration data. **Nothing in the theorem or its proof cares how that region was constructed.** They fill it with a bootstrap from clean data (their Section 3.3) because that is the natural statistical move. It is not the only legal one.

**Verified by reading.** Their supplementary Section A3 was checked. A3.1 is the single-parameter randomised-response model; A3.2 is a two-parameter label-hierarchy extension. Neither is an open clinical prior. Every route in the paper to a valid region requires either M assumed exactly known (Algorithm 1, their example is differential privacy where the parameter is a design choice) or clean data (Algorithms 2/3, including the simplified scalar case in A3.1.4). **There is no third option in the paper.**

**The claim.** Three routes to the matrix exist in the literature, and all three close on mental-health Reddit text:

| Route | Needs | Why it closes |
|---|---|---|
| Clean data (Sesia §3.3) | Verified true labels | Do not exist for bipolar/schizophrenia; DAIC-WoZ is depression-only |
| Anchor points (Xia et al. 2019; Patrini et al. 2017) | Instances where true class is near-certain from features | Bipolar and depression share language by clinical necessity |
| Clusterability (Zhu et al. 2021, HOC) | 2-NN share the *true* class | Fails on lower-quality features; see D-030 |

**And even if one were open:** all three produce a **point estimate**. Theorem 4 needs a **region**. HOC's finite-sample rate (their Theorem 2) is derived under an assumption that off-diagonal entries are *uniform*, which is precisely what this project rejects.

**Consequences.** This is now the central claim. It is structural, cites the threatening papers as support, and is defensible.

---

## D-028 · Elicit in Penso's direction, convert to Sesia's
**Date:** 2026-07-17 · **Status:** ACTIVE · **Category:** C2

**Decision.** Elicit the contamination model as `P(Ỹ = j | Y = i)` (given the true condition, where does the post go?), then convert to Sesia's parameterisation.

**Reasoning.** Sesia's matrix is `P(Y = l | Ỹ = k)`: "given the label says bipolar, what is the chance the truth is depression?" **Nobody can answer that directly**, because it depends on the unknown true prevalence. Penso's direction is the one a clinician can answer. Penso et al. note this explicitly in their Section 4: Sesia and Clarkson "need to know the marginal class frequencies for both the clean and noisy labels, whereas we do not."

**The conversion.** The *noisy* marginals are simply observable (count subreddit memberships). Elicit `P`, observe `ρ̃`, derive `ρ`, then convert by Bayes. No clean data needed anywhere.

**UNVERIFIED.** The specific derivation `ρ = (Pᵀ)⁻¹ρ̃`, and the consequent observation that inconsistent models could be pruned (see D-025, D-031), **is not taken from any published paper.** It is a derivation produced during this session. **Check the algebra and have the supervisor check it before it enters the thesis as a contribution.**

---

## D-029 · Keep Mondrian (label-conditional), not marginal coverage
**Date:** 2026-07-17 · **Status:** ACTIVE · **Category:** C2

**Decision.** C2 targets label-conditional (Mondrian) coverage, giving the guarantee separately per condition.

**Reasoning.** Marginal coverage would let depression's 117k examples drown out bipolar's 5.8k. A system could hit 90% overall while covering bipolar 40% of the time and the headline number would look fine. Per-condition difference **is** the research question.

**The objection to be ready for.** Bortolotti, T., Wang, Y.X.R., Tong, X., Menafoglio, A., Vantini, S. and Sesia, M. (2025) 'Noise-adaptive conformal classification with marginal coverage', arXiv:2501.18060. Same group, extending to marginal coverage. Their results indicate that while all adaptive methods reach valid coverage, **only the marginal ones produce more informative prediction sets**; the label-conditional versions gain validity without gaining informativeness.

**Response.** They optimise for one global guarantee; this project needs per-condition behaviour. **Run a marginal arm alongside to show the contrast rather than avoiding the point.**

**OPEN.** This paper has not been read in full (rate-limited). Their estimation route may be softer on clean data than the 2024 paper, which would pressure D-027. **Read before finalising Chapter 2.**

---

## D-030 · C1 becomes a diagnostic, not an estimator
**Date:** 2026-07-17 · **Status:** ACTIVE · **Category:** C1

**Decision.** C1 stops being "estimate the matrix with Confident Learning plus a clinical prior" and becomes "**demonstrate, on the actual data, that the available estimators fail**." That demonstration is what licenses C2's alternative.

**Why the change was forced.** D-024. HOC estimates the matrix without clean data, so "you need a prior because the data cannot do it" was not established.

**Why the replacement is stronger.** Zhu, Z., Wang, J. and Liu, Y. (2022) 'Beyond images: label noise transition matrix estimation for tasks with lower-quality features', *Proceedings of the 39th ICML*, PMLR 162.

This paper **exists because HOC fails outside computer vision**. Their abstract: tasks with lower-quality features fail to meet the anchor-point or clusterability condition. They test on **BERT-embedded text**, which is this project's exact setting.

Estimation error (x100, lower is better), their Table 2:

| Dataset | Noise | HOC | Confident Learning | T-Revision | Their fix |
|---|---|---|---|---|---|
| AG's News (BERT), 4-class, 30k each | e ≈ 0.178 | 13.32 | 11.41 | 10.38 | 8.35 |
| AG's News | e ≈ 0.302 | 10.62 | 10.63 | 10.71 | 6.52 |
| Jigsaw (BERT), binary, 9.4:1 imbalance | e ≈ 0.111 | **14.25** | **20.17** | **20.92** | 9.97 |
| Jigsaw | e ≈ 0.2 | 11.28 | 16.44 | 17.10 | 7.66 |

**Two things to say out loud.**

1. **AG's News is the closest published analogue** on class count (4) and size (120k vs ~140k). HOC's error is 0.133 while the quantity being estimated has size 0.178. **The error is roughly 75% of the signal.** On *balanced* data with 30,000 per class. This project has 5,780 bipolar.
2. **Jigsaw is the imbalanced one.** At 11% noise, the trivial estimate "assume no noise" scores 11.1. **HOC scores 14.25: worse than assuming the labels are perfect.** Confident Learning scores 20.17, which for binary at that noise level is exactly a random guess.

**And Confident Learning is C1's own engine.** That is not a footnote; it is why C1 changed role.

**The clinical explanation, which is the elegant part.** HOC needs each post's nearest neighbours in embedding space to share its **true** class. A person who truly has bipolar disorder, writing during a depressive episode, writes like a person with depression. That is *phase predominance*, a documented clinical feature. So their nearest neighbours in MentalBERT space *are* depression posts. **The reason clusterability fails is the same clinical fact that creates the label noise.** The estimator's assumption and the phenomenon under study are in direct conflict.

**Also:** HOC's Assumption 2 requires a dominant diagonal. If diagnostic delay means most truly-bipolar people post in r/depression, bipolar's diagonal is not dominant and Theorem 1 does not apply at all.

**What C1 now does.** Run HOC and `cleanlab` on the actual data. Do they disagree? Are they unstable across seeds? Do they produce clinically implausible matrices? **Every outcome is useful:** inside the elicited envelope is independent corroboration; outside is the measured failure the 2022 paper predicts.

---

## D-031 · Do not prune the elicited set (preserve α_V = 0)
**Date:** 2026-07-17 · **Status:** OPEN · **Category:** C2

**Provisional decision.** Option (a) from D-025: do not prune, keep deterministic containment, accept a wider region.

**Why still OPEN.** The trade is real and unresolved. Option (b) gives tighter sets at the cost of `α_V = α_ρ > 0` and loses the corollary that this route is *tighter* than Sesia's bootstrap. With ~140k posts `α_ρ` would be very small. This needs a decision before Chapter 4 is written, ideally with the supervisor.

**Depends on.** D-028's derivation being correct.

---

## D-018 · Two-regime identifiability framing
**Date:** undated · **Status:** ACTIVE, and substantially strengthened 2026-07-17 · **Category:** C1/C3

**Original decision.** Because a plain class-conditional transition matrix is not identifiable from noisy labels alone (Liu, Cheng and Zhang, 2023, ICML), conditions are split into **data-identified** (large support, e.g. depression) and **prior-dominated** (small support, e.g. bipolar and schizophrenia, where the output leans on the clinical prior and is reported as a sensitivity analysis rather than a firm estimate).

**Strengthened 2026-07-17: the split is now arithmetic, not a framing choice.**

Zhu, Song and Liu's Theorem 2 gives HOC's finite-sample rate, but only when the diagonal exceeds a threshold depending on class rarity. At K = 4 with this project's prevalences:

| Condition | Posts | Share | Required diagonal | Achievable? |
|---|---|---|---|---|
| Depression | 117,331 | ~0.81 | 0.27 | **yes, easily** |
| Eating disorder | 14,577 | ~0.10 | 1.00 | no |
| Schizophrenia | 8,712 | ~0.06 | 1.56 | no |
| Bipolar | 5,780 | ~0.04 | **2.25** | **impossible** |

**Depression is the only one of the four for which the leading no-clean-data estimator has a finite-sample guarantee at all.** The other three require a diagonal probability above 1.

**Caveat that must be stated, not hidden.** That threshold comes from their **Theorem 2** (finite-sample rate), derived under a tractability assumption that off-diagonal entries are uniform. Their **Theorem 1** (uniqueness in the infinite-data limit) carries no such condition. So the honest claim is *"HOC has no finite-sample rate for three of four conditions,"* **not** *"HOC provably fails."* It might still work in practice, which is exactly why D-030 runs it and measures.

**UNVERIFIED.** This arithmetic was produced during this session from their formula. **Plug the actual counts in and check it independently.** It is currently the strongest single result in the project, which means it is also the one that will hurt most if it is wrong.

**A further refinement.** The regime boundary is **not just sample size**. Zhu, Wang and Liu (2022) show AG's News has 30,000 examples per class, perfectly balanced, and HOC still errs by 0.133 because BERT text features are not clusterable enough. So the boundary is set by **sample size AND feature quality together**. For the rare conditions here, both are against the project at once. This is a sharper version of the original thesis than the one it started with.

**Amendment (2026-08-14): provenance of the four counts, and why two totals exist.**

The counts in the table above come from `Docs/project-proposal.md`, in the "Dataset correction (June 2026)" framing note, repeated in its dataset table (§5) and its timeline (§8):

> "all four core conditions have been downloaded and directly verified (June 2026): **depression (117,331 posts), eating disorder (14,577), schizophrenia (8,712), bipolar (5,780)**"

They sum to 146,400. That proposal is being relocated out of the repository, so the figures and their source are recorded here to keep the paper trail with the arithmetic that uses them.

`python -m src.data` reports 140,086 rows, which reads as a contradiction of the proposal. It is not one. Both were re-measured on 2026-08-14 and both are correct: they count different things.

| Condition | Raw rows on disk | After dedup | Removed | Share (raw) | Share (dedup) |
|---|---|---|---|---|---|
| Depression | 117,331 | 113,576 | 3,755 (3.2%) | 0.8014 | 0.8118 |
| Eating disorder | 14,577 | 13,447 | 1,130 (7.8%) | 0.0996 | 0.0961 |
| Schizophrenia | 8,712 | 7,680 | 1,032 (11.9%) | 0.0595 | 0.0549 |
| Bipolar | 5,780 | 5,195 | 585 (10.1%) | 0.0395 | 0.0371 |
| **Total** | **146,400** | **139,898** | **6,502 (4.4%)** | | |

The 15 Zenodo CSVs hold exactly 146,400 rows, reproducing the June 2026 verification file for file, and no row is dropped as an unmapped subreddit. `combine_sources()` then removes 6,502 rows that are exact duplicates on `(text, source, author_id)` (`src/data/combine.py`), giving 139,898 Reddit rows; the 140,086 total adds DAIC-WoZ's 188 participants. So the proposal counted the files and the loader counts the corpus, and nothing on disk has changed since June.

**Which denominator this arithmetic should use.** The estimator sees the deduplicated corpus, not the raw files, so the shares should be the post-dedup column. The correction runs against the rare conditions: deduplication removes 3.2% of depression but 7.8% to 11.9% of the other three, so every rare class becomes rarer (bipolar moves from 0.0395 to 0.0371). In the table above the required diagonal falls as class share rises, so recomputing on post-dedup shares pushes the three rare conditions' thresholds further above 1 rather than below. The direction of the correction is therefore safe. That is an observation about the monotonicity of the table, not a re-derivation of it.

**The UNVERIFIED flag above stands.** Only the provenance half of open item 4 is closed by this amendment. The formula itself, taken from Zhu, Song and Liu's Theorem 2, has still not been checked independently, and these counts should be plugged into it when it is. One refinement to make at that point: HOC ran on the training split (111,892 rows), not the full corpus, so the operative shares may be the train-split ones rather than either column here.

**Links.** Depends on D-030 (C1 measures rather than assumes). Feeds open item 4. Records provenance held in `Docs/project-proposal.md` before that file leaves the repository.

---

## D-032 · Three-arm experiment as the core evaluation
**Date:** 2026-07-17 · **Status:** ACTIVE · **Category:** C2/C3

**Decision.** The headline evaluation compares three arms:

| Arm | Description |
|---|---|
| **(a)** | Noise-blind: standard conformal, ignoring label noise |
| **(b)** | Sesia's Algorithm 2 with a plug-in estimate (from HOC or cleanlab) |
| **(c)** | This project: Sesia's Algorithm 2 with the clinically elicited region |

**Reasoning.** Arm (b) is essential. Beating (a) is table stakes and proves nothing anyone will care about. The claim reduces to (c) beating (b) on risk-coverage for bipolar and schizophrenia.

**Cost.** Low. Sesia's code is public at github.com/msesia/conformal-label-noise; HOC's at github.com/UCSC-REAL/HOC.

**The insurance, and why the project is safe either way.** Add a **noise-heterogeneity sweep**: vary how unequal the true noise rates are synthetically and find where (c) overtakes (b), then locate the real data on that axis. Head-to-head, (c) > (b) is a coin flip that can be lost. With the sweep, losing is still a result: *"the coupling pays off above heterogeneity level h*, and Reddit proxy noise sits below it"* is a publishable, honest finding.

**The same logic applies to the feasibility threshold.** If C2 works, the project has a method. If it degenerates to always abstaining on bipolar, the project has **the first characterisation of where this family of methods stops working on real mental-health proxy labels**. A project whose headline survives its own negative result is well designed, and that should be said in the defence.

---

## D-034 · Clusterability diagnostic results: the prediction was falsified
**Date:** 2026-07-24 · **Status:** ACTIVE · **Category:** C1

**The prediction.** Before running, the stated expectation was: 2-NN agreement
would be high for depression and noticeably low for bipolar, and low bipolar
agreement would be evidence that HOC's clusterability condition fails on this
data. It was explicitly framed as a one-sided test: high bipolar agreement would
falsify the clusterability-failure argument.

**The result.** Run on MentalBERT embeddings from the Milestone 0 fine-tuned
checkpoint. Reddit training split, 111,892 posts. |E| = 15,000, G = 20 rounds,
negative cosine similarity, seed 0.

| Condition | n/round | per-neighbour agreement | sd | chance | lift over chance |
|---|---|---|---|---|---|
| depression | 12,173 | 99.54% | 0.07% | 81.2% | 1.23x |
| eating_disorder | 1,438 | 98.78% | 0.39% | 9.6% | 10.30x |
| schizophrenia | 822 | 94.49% | 0.74% | 5.5% | 17.22x |
| bipolar | 565 | **86.70%** | 1.10% | 3.8% | **23.00x** |

**The prediction was wrong.** Bipolar agreement is 86.70%, inside and near the
top of the 78 to 88 percent range Zhu, Song and Liu report for noisy CIFAR-10
(their Table 3), a setting where HOC works. Bipolar also has the **highest** lift
over chance of all four conditions at 23x, meaning bipolar posts are the most
distinctive class in this embedding space relative to their base rate, not the
least.

**Same-author contamination ruled out.** Nearest neighbours share the centre
post's author in under 0.1% of neighbour slots across all conditions (bipolar
0.08%, depression 0.01%). Authors average roughly 1.1 posts each in the training
split. The observed clustering is not an artifact of near-duplicate posts by the
same person.

**Neighbour label distribution given centre label** (row-normalised):

| centre \ neighbour | bipolar | depression | eating_dis. | schizophrenia |
|---|---|---|---|---|
| bipolar | 0.8669 | **0.0995** | 0.0027 | 0.0310 |
| depression | 0.0029 | 0.9954 | 0.0006 | 0.0011 |
| eating_disorder | 0.0012 | 0.0101 | 0.9878 | 0.0008 |
| schizophrenia | 0.0205 | **0.0324** | 0.0020 | 0.9450 |

**One positive finding.** For both bipolar and schizophrenia centres, the
dominant off-diagonal neighbour class is depression (9.95% and 3.24%
respectively). This is directionally consistent with the clinical elicitation
argument (bipolar-to-depression is the expected dominant confusion flow). It is
weak evidence: it concerns embedding neighbourhoods rather than label noise
directly, and the depression class is also the largest by a wide margin. Record
it as consistent-with, not as support.

**Two reasons the falsification is weaker than it first appears.**

1. **Circular measurement.** The embeddings come from a model fine-tuned to
   separate these four noisy labels. Agreement on noisy labels was measured in a
   space optimised to separate noisy labels. Depression at 99.54% against Zhu et
   al.'s 78 to 88 percent is the tell: the gap is the training objective showing
   through. Zhu et al. also use noisy-label-trained extractors, but measure
   feasibility against **true** labels, which breaks the circularity. That option
   is not available here.

2. **Structural blindness to the failure mode of interest.** Posts are
   partitioned by their *noisy* label. A truly-bipolar person who posted in
   r/depression sits in the depression row, indistinguishable from truly-depressed
   posts. The bipolar row therefore measures only the self-identified subset,
   those whose proxy label happened to be correct. The phase-predominance cases
   the argument depends on are filed elsewhere by construction.

**Integrity note, recorded deliberately.** Reason (2) is true a priori and should
have been identified when the test was designed. It was articulated only after
the result came back against the hypothesis. That is the structural shape of
motivated reasoning, and it is logged as such so that a reader can discount it
appropriately. The point stands or falls on its own merits, not on the timing of
its appearance.

**Consequence.** The claim that clusterability fails on this data currently has
**no support from this project's own data.** It rests entirely on Zhu, Wang and
Liu's (2022) published Table 2 results on other BERT corpora. That remains real
published evidence from the method's own authors, but it is borrowed rather than
measured.

**What this promotes.** Stage 2 becomes the load-bearing diagnostic: HOC's
stability across random seeds, and HOC-versus-cleanlab disagreement. Neither
carries the circularity confound. **If HOC returns a stable and clinically
plausible matrix on this data, D-030's premise is in trouble and C1 needs
rethinking again.**

**Links.** Tests the assumption behind D-030. Supersedes the framing in open
item 2.

---

## D-035 · Base-embedding control run
**Date:** 2026-07-24 · **Status:** ACTIVE · **Category:** C1

**Decision.** Repeat the D-034 diagnostic using embeddings from **base
MentalBERT** (`mental/mental-bert-base-uncased`, no fine-tuning on this
project's four labels), as a control against the fine-tuned run.

**What this does and does not establish.** It does **not** produce "the real
clusterability number." Clusterability is defined over true labels, which are
unavailable, so it cannot be measured directly by any run. What the control
provides is the **delta**: the difference between fine-tuned and base agreement
quantifies how much of the apparent structure in D-034 is an artifact of the
training objective rather than intrinsic to the text.

**Interpretation rule set in advance, to avoid post-hoc reasoning.**

- If base bipolar agreement stays high (say above 70%), the clusterability-failure
  argument is genuinely weak and should be **dropped rather than defended**. The
  C1 justification then rests solely on Zhu, Wang and Liu (2022) plus whatever
  Stage 2 shows.
- If base bipolar agreement drops sharply (say below 40%), the fine-tuned number
  was substantially artifact, and the honest report is that neither number
  resolves clusterability but the space is far less separable than D-034 implied.

**What this does NOT change.** HOC itself should still be run on the
**fine-tuned** embeddings, because that matches HOC's own protocol: Zhu, Song and
Liu take the feature extractor from a model trained to near-100% training
accuracy on the noisy labels. Running HOC on base embeddings would handicap it
unfairly and would not be a fair test of the method.

**Links.** Controls for the confound identified in D-034. D-040 adds a separate,
sanctioned arm that runs HOC on non-fine-tuned extractors; it asks a different
question (does the failure survive the prescribed remedy?) and does **not** reverse
the protocol decision recorded here.

---

## D-036 · Base-embedding control result, and pre-registration for Stage 2
**Date:** 2026-07-24 · **Status:** ACTIVE · **Category:** C1

**The pre-registered rule (D-035) fired.** D-035 set the interpretation in
advance: base bipolar agreement above roughly 70% means the clusterability
argument should be dropped; below roughly 40% means the fine-tuned number in
D-034 was substantially artifact. Base bipolar came in at 24 to 31% depending on
neighbour search scope (see below). The rule fires on the second branch, on a
threshold set before the data existed.

**Result.** Base MentalBERT (`mental/mental-bert-base-uncased`, no fine-tuning on
this project's labels), same protocol as D-034: |E| = 15,000, G = 20, negative
cosine similarity.

| Condition | base | fine-tuned | delta | base chance | base lift |
|---|---|---|---|---|---|
| depression | 94.5% | 99.5% | 5.0pp | 81.2% | 1.16x |
| eating_disorder | 71.1% | 98.7% | 27.6pp | 9.6% | 7.45x |
| schizophrenia | 41.2% | 94.6% | 53.4pp | 5.5% | 7.53x |
| bipolar | **29.3%** | **86.7%** | **57.4pp** | 3.8% | 7.77x |

**The pattern is the finding.** Fine-tuning moved depression by 5pp and bipolar
by 57pp. It did almost nothing for the class that was already separable and
manufactured nearly all of the apparent structure for the rare ones. In base
space all three rare conditions sit at a similar lift over chance (roughly 7.5x),
meaning the *differential* clusterability seen in D-034 was an artifact of the
training objective rather than a property of the text.

**Base neighbour label distribution given centre label:**

| centre \ neighbour | bipolar | depression | eating_dis. | schizophrenia |
|---|---|---|---|---|
| bipolar | 0.2445 | 0.6318 | 0.0299 | 0.0938 |
| schizophrenia | 0.0915 | 0.5232 | 0.0296 | 0.3556 |

In label-agnostic space, the majority of a bipolar post's nearest neighbours
carry the depression label. **Caveat that must be stated in the thesis:**
depression is 81.2% of the pool, so 63% is *below* chance. The honest reading is
that bipolar posts cluster with bipolar at roughly 6.5x chance, but in absolute
terms most of their neighbours are still depression posts. HOC's condition is
about **absolute** clusterability, not lift, so the relevant figure is 24 to 31%
against the 78 to 88% range where Zhu, Song and Liu validate the method.

---

**Neighbour search scope: diagnosis of a reproduction discrepancy.**

An independent reproduction agreed with the pipeline to within 0.1pp on the
fine-tuned features but came in roughly 5pp lower on the three rare conditions in
base space. The cause was initially attributed to fp16 precision. **That was
wrong.** The cause is neighbour search scope: the pipeline searches for 2-NN over
the **full dataset** (111,892 posts), while the reproduction searched **within the
sampled subset E** (15,000 posts). Measured directly, holding everything else
fixed:

| Condition | within-E | full-dataset | scope effect |
|---|---|---|---|
| bipolar | 26.9% | 30.9% | +4.0pp |
| depression | 94.2% | 94.7% | +0.5pp |
| eating_disorder | 66.9% | 72.3% | +5.4pp |
| schizophrenia | 33.5% | 39.9% | +6.4pp |

This reproduces the observed discrepancy in both sign and magnitude.

**Which scope is correct.** HOC searches within E, deliberately:
`n1 = arg min over n' in E, n' != n`. Zhu, Song and Liu restrict it to preserve
the i.i.d. property of the 3-tuples so the consensus estimates stay consistent,
and go further with their E*_3 disjointness condition. For a diagnostic intended
to characterise HOC's operating regime, **within-E is the correct scope**, and the
pipeline as originally written measures something slightly easier than what HOC
faces. Both are reported; within-E is primary.

**A finding inside that discrepancy.** The scope change costs 4 to 6pp on the
rare conditions but only 0.5pp on depression. At |E| = 15,000, a bipolar post has
roughly 566 same-class candidates to match against, versus 4,221 in the full
data. **The rare conditions are neighbour-starved in exactly the regime HOC
operates in.** This is an independent difficulty, separate from clusterability
itself, and belongs in the C3 limits analysis.

**On fp16, since it was the original hypothesis.** Median gap between the 2nd and
3rd nearest similarity, against an fp16 dot-product error scale of roughly 1e-3
for 768-dimensional unit vectors:

| Space | bipolar | depression | eating_dis. | schizophrenia |
|---|---|---|---|---|
| base | 0.00204 | 0.00187 | 0.00329 | 0.00286 |
| fine-tuned | 0.00049 | 0.00029 | 0.00014 | 0.00023 |

Counterintuitively the **fine-tuned** space is the fp16-vulnerable one: its median
gaps sit below the error scale, so neighbour ordering there is substantially
numerical noise. It does not materially affect the agreement statistic, because
in that space almost every near-neighbour carries the same label anyway, so
reordering near-ties rarely changes the outcome. This is why the two runs matched
to 0.1pp on fine-tuned features. In base space the gaps are 2 to 3x the error
scale, so less reordering occurs, but labels are heterogeneous so any reordering
does move the number. A single fp32 pass is scheduled to bound the effect.

---

**The sharpened argument this enables.** HOC estimates T from the frequency of
*disagreement* among neighbours' noisy labels; its consensus equations fit T to
observed agreement patterns. If the feature extractor has been trained to make
those labels agree, the disagreement signal is suppressed and HOC will fit a T
close to identity.

Zhu et al. do not encounter this on CIFAR because their noise is **synthetic and
feature-independent**: the true class structure remains in the image regardless
of the label flip. This project's noise is **feature-dependent**. A truly-bipolar
person writing during a depressive episode produces text that genuinely resembles
depression, so a model fine-tuned on proxy labels learns to separate the
self-selection behaviour rather than the condition. The measurement above
supports this: fine-tuning created 57pp of bipolar clusterability that was not
present in the text.

The gap-distribution table adds independent support. In fine-tuned space the
classes have collapsed into blobs where within-cluster distances are near-uniform
(median gaps of 0.0001 to 0.0005), so which two neighbours HOC selects is close to
arbitrary-within-class. The consensus patterns it counts will therefore be
overwhelmingly within-class agreement.

This is an argument with measurement behind it, not a proof. Stage 2 is what
would convert it.

---

**PRE-REGISTERED PREDICTION FOR STAGE 2.** Recorded before the HOC run, so the
interpretation cannot be fitted to the result. This is the discipline whose
absence is recorded in D-034.

*Prediction:* HOC run on the **fine-tuned** embeddings returns diagonal entries
above roughly 0.85 for all four conditions, implying substantially less label
noise than the clinical literature suggests for bipolar, because the disagreement
signal the estimator depends on has been suppressed by the fine-tuning.

*Falsifier:* HOC returns a bipolar diagonal below roughly 0.7, with a dominant
bipolar-to-depression off-diagonal, stable across random seeds. If that occurs,
HOC is working on this data and the clusterability line of argument should be
dropped entirely rather than defended.

*Fallback if falsified:* the C1 argument does not depend on HOC failing. Even a
well-behaved HOC produces a **point estimate**, while Sesia's Theorem 4 consumes
a **confidence region**, and HOC's own finite-sample rate (their Theorem 2) is
derived under a uniform-off-diagonal assumption that is unsatisfiable at three of
the four class prevalences here (see D-018). That argument concerns what
Algorithm 2 requires as input, not estimation quality, and survives either
outcome.

*Consistency check available:* Zhu, Wang and Liu (2022) report HOC scoring 14.25
on Jigsaw where the trivial identity assumption scores 11.1. HOC being worse than
assuming no noise is what a suppressed-disagreement failure looks like.

**Links.** Fires the D-035 pre-registered rule. Sets the pre-registration for
Stage 2 (`src/noise/hoc_estimate.py`). The scope correction is implemented in
`src/noise/clusterability.py` per this entry.

---

## D-037 · HOC full result: the pre-registered prediction held
**Date:** 2026-07-24 · **Status:** ACTIVE · **Category:** C1

**The D-036 prediction held.** HOC on the fine-tuned embeddings returned diagonal
entries above 0.85 for all four conditions. The falsifier (bipolar diagonal below
0.7 with a dominant bipolar-to-depression off-diagonal) did not occur, though the
direction of the top off-diagonal is bipolar-to-depression as the clinical
literature predicts. The prediction was recorded in D-036 before this run.

**Mean transition matrix (5 seeds):**

| true \ noisy | bipolar | depression | eating_dis. | schizophrenia |
|---|---|---|---|---|
| bipolar | **0.9534** | 0.0263 | 0.0035 | 0.0168 |
| depression | 0.0025 | 0.9964 | 0.0003 | 0.0008 |
| eating_disorder | 0.0008 | 0.0039 | 0.9946 | 0.0007 |
| schizophrenia | 0.0133 | 0.0074 | 0.0012 | 0.9781 |

HOC estimates bipolar label noise at 4.7%, for the condition the clinical
literature identifies as the most mislabelled of the four.

**The estimator is internally healthy by every check, which is the point.**

- **Stable across seeds.** Per-seed bipolar diagonal: 0.9546, 0.9526, 0.9540,
  0.9525, 0.9533. Standard deviation 0.0008. This is not a wobbly estimate that
  more rounds would settle; it is tight and consistent.
- **Assumptions satisfied.** All five seeds produced a non-singular matrix
  (determinant 0.922 to 0.925) and every row is diagonally dominant. HOC's own
  Assumption 1 (invertibility) and Assumption 2 (diagonal dominance) hold, so the
  result cannot be dismissed on the grounds that its preconditions failed.
- **Prior recovered almost exactly.** Estimated class prior matches the observed
  noisy proportions to an L1 divergence of 0.004 (bipolar 0.036 vs 0.038,
  depression 0.814 vs 0.812, and so on). The estimator is fitting the data it is
  given correctly.

**Why this is the strongest form of the C1 result.** By every internal diagnostic
HOC is behaving properly: stable, invertible, diagonally dominant, prior
recovered. And it returns a bipolar noise rate of 4.7%. This is precisely the
"clean, confident, no warning sign" failure predicted in D-036. There is no check
a practitioner could run on HOC's output that would reveal a problem. An estimator
that were merely broken could be distrusted in general; a competent estimator that
returns an implausible answer with no signal of trouble is the sharper and more
dangerous case, and it is the one measured here.

**Correlation with the fine-tuned 2-NN artifact.** HOC's diagonal tracks the
fine-tuned 2-NN agreement statistic from D-034 at Pearson 0.999, rank order
identical. This is expected from the mechanism (HOC fits its matrix to 2-NN
consensus frequencies) and is what converts the correlation from coincidence into
confirmation: the estimator's output is a deterministic function of a statistic
that D-036 established is largely a fine-tuning artifact.

**Load-bearing argument, restated.** Do not lead with "4.7% is clinically
implausible," which is a judgement. Lead with: HOC's output is a deterministic
function of the 2-NN consensus structure, that structure was measured (D-034,
D-036) to be substantially manufactured by fine-tuning on the labels being
estimated, therefore whatever the true noise rate is, HOC cannot be measuring it
here because its input was manufactured. The clinical implausibility is
corroboration, not the load-bearing claim.

**Reproduction note.** This run used fresh explicit seeds and reproduced the
original mean matrix (hoc_mean_T.csv) to within 5e-4, which is Monte Carlo
variation in the neighbour sampling, consistent with the per-seed spread of the
same order. The two HOC runs are the same result, not conflicting results.

**Consequence.** The HOC half of the C1 diagnostic is complete and rests on
measured evidence from this project's own data. Remaining: the cleanlab
(Confident Learning) estimate, for a second independent estimator. If cleanlab
also returns a near-identity bipolar row, two independent estimators fail
identically on this data, which is the strongest form of the result.

**Links.** Confirms the prediction in D-036. Completes the HOC portion of D-030.

---

## D-038 · Cleanlab cross-validation fold integrity verified
**Date:** 2026-07-24 · **Status:** ACTIVE · **Category:** C1

**Decision context.** The cleanlab (Confident Learning) estimate is the second
independent estimator in the C1 diagnostic. Its out-of-sample probabilities are
only valid if author-grouping holds across folds (D-010), because author style
leakage would inflate the probabilities and silently corrupt the estimated
transition matrix.

**Verification.** The fold assignment was checked directly. Across 111,892 posts
and 99,749 unique authors, zero authors appear in more than one fold. Per-fold
condition proportions match within roughly half a percentage point (bipolar
0.0372 to 0.0380, depression 0.8092 to 0.8144). Three folds, evenly sized at
~37,300 posts each. Posts per author average 1.12, maximum 8.

**Consequence.** The cleanlab out-of-sample probabilities are not contaminated by
author leakage. Whatever transition matrix cleanlab returns can be compared
against the HOC result (D-037) as a genuinely independent estimate. The mandatory
group-integrity test (D-010) passed.

**Links.** Depends on D-010. Sets up the cleanlab-vs-HOC comparison that completes
the C1 diagnostic.

---

## D-039 · Cleanlab result: agrees with HOC on structure, contradicts on magnitude
**Date:** 2026-07-24 · **Status:** ACTIVE · **Category:** C1

**Result.** Cleanlab (Confident Learning) on author-grouped out-of-sample
probabilities (folds verified clean, D-038) returned:

| true \ noisy | bipolar | depression | eating_dis. | schizophrenia | implied noise |
|---|---|---|---|---|---|
| bipolar | 0.8483 | 0.1027 | 0.0062 | 0.0429 | 15.2% |
| depression | 0.0093 | 0.9790 | 0.0041 | 0.0076 | 2.1% |
| eating_disorder | 0.0017 | 0.0277 | 0.9685 | 0.0022 | 3.2% |
| schizophrenia | 0.0326 | 0.0638 | 0.0048 | 0.8988 | 10.1% |

**Agreement with HOC (D-037), on structure.** Both estimators rank the conditions
identically (depression most reliable, then eating disorder, schizophrenia,
bipolar). Diagonal Pearson correlation 0.982. Both identify bipolar as the
noisiest condition and both place bipolar's dominant off-diagonal leak into
depression, the clinically predicted direction.

**Disagreement with HOC, on magnitude.** Cleanlab estimates roughly three times
more noise throughout. Bipolar: HOC 4.7 percent, cleanlab 15.2 percent.
Schizophrenia: HOC 2.2 percent, cleanlab 10.1 percent. HOC's estimate carries a
tight cross-seed confidence region (bipolar diagonal std 0.0008) that would
exclude cleanlab's estimate; cleanlab provides a single point with no region.

**Interpretation, stated carefully.** This is not the "both estimators flatline to
near-identity" outcome. Cleanlab's 15 percent for bipolar is more clinically
plausible than HOC's 4.7 percent. The C1 claim is therefore not "the estimators
agree on an implausible answer." It is the sharper and more defensible claim that
two independent data-driven estimators, both internally healthy and both producing
confident output with no error signal, disagree by a factor of three on the noise
magnitude, and there is no data-internal way to adjudicate between them. Neither
supplies the confidence region that Sesia's Theorem 4 requires, and they are
mutually incompatible on where that region would sit. This is the situation that
motivates supplying the region from clinical knowledge (C2).

**Anticipated objection.** Since cleanlab's bipolar estimate is closer to
clinically plausible, does data-driven estimation "work" after all? No: cleanlab
gives a point estimate with no region, HOC gives a tight region that excludes it,
and nothing internal to either method indicates which is correct. Confident mutual
contradiction is a failure to identify the matrix, not a success.

**Links.** Compares against D-037. Depends on D-038. Completes the two-estimator
C1 diagnostic.

---

## D-040 · HOC on representations not trained on the project's labels (PRE-REGISTRATION)
**Date:** 2026-08-09 · **Status:** ACTIVE · **Category:** C1

**Decision.** An additional two-arm HOC run was pre-registered, on feature
extractors that have never been trained on this project's noisy proxy labels, to
test whether the D-037 result survives the remedy the identifiability literature
prescribes for exactly that failure.

**Everything below this line was written before any Arm A or Arm B number was
produced, and committed before the first estimation was run.** That ordering is
the reason D-036 and D-037 are defensible, and it is the reason this entry exists
as a separate record rather than as a preamble to its own results (which are in
D-041).

---

### The question

Does HOC's output on this data change materially when the feature extractor is
**not** trained on the project's noisy proxy labels?

### Context: a fourth identifiability route this project had not addressed

Liu, Cheng and Zhang (2023, ICML, PMLR 202) has now been read in full. It
documents **four** routes to an identifiable noise-transition matrix, not the
three this project had been working from: multiple conditionally independent noisy
labels, anchor points, clusterability, and **disentangled informative features**
(their Theorem 5.5). The fourth was unaddressed here.

Their Section 6 tests it. Table 1 estimates T on CIFAR-10 with the HOC estimator
over three feature extractors, and the ordering is why this matters:

| Encoder | asymm. 0.3 error | asymm. 0.4 error |
|---|---|---|
| Weakly supervised (cross-entropy on the noisy labels) | 14.51 | 15.2 |
| SimCLR (self-supervised) | 4.42 | 4.41 |
| IPIRM (disentangled) | 3.73 | 3.74 |

They describe the first row as the standard protocol used by forward correction
and by HOC itself, and note that because its training data is noisy there is no
guarantee the resulting features are disentangled. **That first row is D-035's
protocol, and it is the worst of the three.**

### Relationship to D-035: this is an additional arm, not a reversal

D-035 states that HOC must run on fine-tuned embeddings because that matches
HOC's own published protocol, and that running it on base embeddings would
handicap the method unfairly. That reasoning is sound **for the question D-035
asks**, which is whether HOC-as-published works on this data. D-037 answered that
question.

It does not settle a different question: whether HOC's failure **survives the
remedy the identifiability literature prescribes for it**. D-035 stays ACTIVE and
its text is unchanged. D-040 is a distinct experiment with a distinct question,
and D-041 will report it as such.

### The arms

Both arms run the **identical HOC configuration used for D-037**, verified against
the notebook cell that produced it: seeds `0 1 2 3 4`, `n_rounds = 50`,
`sample_size = 15000`, `max_iter = 1500`, `lr = 0.1`, cosine metric, within-E
neighbour scope. A configuration difference between arms would make the
comparison uninterpretable, so no CLI default is accepted without checking it
matches. Embeddings are cached at fp16 in every arm, as the D-034 and D-036 caches
already are.

| Arm | Extractor | Trained on this project's labels? | Analogue in Liu et al. Table 1 |
|---|---|---|---|
| **Fine-tuned (D-037, already run)** | Milestone 0 MentalBERT checkpoint | yes | row 1, weakly supervised |
| **A** | Base MentalBERT, `mental/mental-bert-base-uncased` | no | none exactly; masked-LM pretraining only |
| **B** | `sentence-transformers/all-mpnet-base-v2` | no | approximately row 2, contrastive |

Arm A is free: the base cache already exists at `embeddings/train__base` from the
D-036 control run, so it needs only a HOC invocation. Arm B needs a fresh
extraction pass (`--extractor mpnet`, cache `embeddings/train__mpnet`).

### The prediction, stated numerically

**Anchor.** D-037 measured a bipolar diagonal of **0.9534** with a cross-seed
standard deviation of **0.0008** on the fine-tuned features, `det(T)` between
0.922 and 0.925, and every row diagonally dominant on all five seeds.

**Predicted.** The bipolar diagonal **falls below 0.70 on Arm A**, most likely
into the range **0.05 to 0.45**, and `det(T)` degrades well below D-037's 0.92.
Arm B is predicted to land **between Arm A and the fine-tuned value, and below
0.85**. The arm-wise Pearson correlation between the HOC diagonal and the 2-NN
agreement statistic on the same features is predicted to **stay above 0.9**,
because the prediction rests on that coupling being mechanistic.

**Why this is the prediction the project's own mechanism licenses.** D-037
recorded that HOC's diagonal tracks the 2-NN agreement statistic at Pearson 0.999.
D-036 measured within-E base bipolar 2-NN agreement at **26.9%** against 86.7%
fine-tuned. If the coupling is mechanistic rather than coincidental, the base
diagonal must fall, and by a lot. There is a degenerate limit worth naming in
advance: if the base-space consensus approaches independence, so that the
empirical `c2` approaches the outer product `c1 ⊗ c1`, then a consensus-matching
optimum is every row of T equal to the noisy marginal, which puts the bipolar
diagonal near **0.038** and `det(T)` near **zero**.

**This prediction runs against the outcome that would most help C1.** The branch
predicted below is NARROWED or DEGENERATE, not STRENGTHENED. Recording it that way
is deliberate: predicting the convenient outcome would make a confirmation
worthless, and D-034 is on record as the entry where an inconvenient result was
reported rather than defended.

### The decision rule, set in advance

All thresholds below are fixed as of this entry. `0.85` and `0.70` are carried
over unchanged from D-036's prediction and falsifier so the family of thresholds
stays consistent across the C1 entries rather than being retuned per run.
"Non-singular" for the purposes of this rule means **|det(T)| >= 0.5**, a
judgement set in advance (D-037 sits at 0.92, a matrix with identical rows sits at
0). Note that `HOCResult.is_nonsingular` uses a looser `1e-8` numerical guard;
that flag is a crash detector, not this rule's threshold. "Diagonally dominant"
means every row satisfies `T[i][i] > 0.5`, which is HOC's own Assumption 2 as
already implemented.

| Branch | Fires when | Consequence |
|---|---|---|
| **STRENGTHENED** | bipolar diagonal **>= 0.85 on every** non-fine-tuned arm, **and** that arm is non-singular, **and** every row is diagonally dominant | The estimator's implausible answer is not an artefact of fine-tuning alone. C1's claim is strengthened and generalised: the failure survives the prescribed fix. |
| **NARROWED** | bipolar diagonal **< 0.70 on any** arm, **and** that arm is non-singular, **and** every row is diagonally dominant | The fine-tuning-artefact explanation is sufficient. C1's claim narrows to fine-tuned representations specifically, and Chapter 1 must be amended to say so. |
| **DEGENERATE** | bipolar diagonal **< 0.70** but Assumption 1 or Assumption 2 **fails** on that arm | Estimator collapse, not rescue. Neither of the two branches above fires. Reported as HOC being unable to return a usable matrix on that representation. |
| **INDETERMINATE** | any arm lands in **[0.70, 0.85]** | Reported as a number, with no strengthening or narrowing claim attached to it. |
| **DISAGREE** | one arm **>= 0.85** and another **< 0.70** | The third outcome. Reported as disagreement between representations, **not** resolved by picking the more convenient arm. |

**Why the DEGENERATE branch exists.** A low bipolar diagonal is ambiguous between
two opposite readings: HOC recovering real noise once the suppression is removed,
or HOC collapsing because the consensus signal has gone. Those look identical in
the diagonal alone and are separated by `det(T)` and by row diagonal dominance.
Without this gate the rule would read a collapse as a rescue. The distinction is
drawn here, before the numbers exist, precisely because it would be unconvincing
if drawn afterwards.

### The limitation, recorded before the result is known

Liu et al.'s **Definition 5.3** defines disentanglement as the feature components
being conditionally independent **given the true label Y**. The true labels are
unavailable on this data, so this property **cannot be verified here by any run**,
in exactly the way D-035 already records that clusterability cannot be measured
directly.

**Arm B therefore does not test Theorem 5.5's hypothesis.** It tests a weaker,
observable proxy: whether HOC's output changes when the extractor is independent
of the project's noisy labels and was trained with an objective that encourages
semantic separation. Additionally, `all-mpnet-base-v2` is trained on large-scale
sentence-pair data, which is not purely self-supervised in SimCLR's sense, so the
analogy to Liu et al.'s middle row is **approximate**, not exact.

**The claim this experiment can support:** that the failure persists across
representations which differ in how they were supervised.
**The claim it cannot support:** that a fully disentangled representation would
not rescue the estimator.

That wording is to be carried into the thesis verbatim rather than softened.

### Confound bounding what any arm can show

`sample_size` is held fixed at 15,000 across arms, so the neighbour starvation
recorded in D-036 is **identical across arms** and therefore does not invalidate
the comparison. It does bound what any arm can show: at |E| = 15,000 a bipolar
post has roughly 566 same-class candidates to match against, versus roughly 4,221
in the full split. This is to be stated in D-041 rather than discovered later.

### Consequences

- Commits the project to reporting D-041 whichever branch fires, including the two
  branches that cost C1 something.
- Adds a third feature extractor to `src/noise/embeddings.py` and therefore a
  third embedding cache. The existing D-034 and D-036 caches are untouched.
- Promotes Liu, Cheng and Zhang (2023) from the "cited but not read in full" table
  to the verified table, per D-033.
- Per-seed spread and estimated prior p are written per arm by the existing
  `per_seed_frame` path, so open item 1's request for the spread is closed for
  every arm rather than for the fine-tuned one only.

**Links.** Adds an arm alongside D-035, which stays ACTIVE and unreversed. Builds
on D-036 (the artefact measurement), D-037 (the fine-tuned HOC result and the
Pearson 0.999 coupling) and D-039 (the second estimator). Results in D-041.
Applies D-033.

---

## D-041 · D-040 result: HOC's structure is stable, its magnitude is representation-dependent
**Date:** 2026-08-09 · **Status:** ACTIVE · **Category:** C1

**Result in one line.** Neither of D-040's headline branches fired. HOC's estimated
bipolar noise rate moves from 4.6% to 45.9% purely as a function of which encoder
produces the features, while the *ordering* of the four conditions and the estimator's
internal confidence stay unchanged. The matrix is not identified; the ranking is.

**Runs.** Both arms executed 2026-08-09 on Colab, 104 minutes total, using D-037's
configuration verbatim (seeds 0 to 4, G = 50, |E| = 15000, max_iter = 1500, lr = 0.1,
cosine, within-E). Arm A reused the D-036 base cache; Arm B extracted fresh into
`embeddings/train__mpnet`.

---

### The pre-registered prediction, item by item

D-040 recorded these before the run. Reproduced here verbatim in substance, with the
verdict against each.

| Prediction (D-040) | Outcome | Verdict |
|---|---|---|
| Arm A bipolar diagonal falls below 0.70 | 0.5409 | **held** |
| Most likely into 0.05 to 0.45 | 0.5409 | **missed**, the drop was real but less severe than predicted |
| Arm A `det(T)` degrades well below D-037's 0.92 | 0.2595 | **held** |
| Arm B lands between Arm A and fine-tuned, and below 0.85 | 0.5409 < 0.7071 < 0.9534 | **held exactly** |
| Arm-wise Pearson between HOC diagonal and 2-NN agreement stays above 0.9 | 0.9940 and 0.9952 | **held** |
| Predicted branch for Arm A: NARROWED or DEGENERATE | DEGENERATE | **held** |

**The prediction was directionally right and quantitatively too pessimistic.** The
degenerate-limit reasoning in D-040 (if base-space consensus approaches independence,
every row of T collapses to the noisy marginal and the bipolar diagonal approaches
0.038) described the direction of travel correctly but overshot the distance. Arm A
is part-way to that limit, not at it. Recording the miss matters more than the hit:
the 0.05 to 0.45 range was reasoned from a mechanism, and the mechanism is only
partially right.

### Branches

| Arm | Extractor | Bipolar diagonal | Cross-seed sd | Mean \|det\| | Branch |
|---|---|---|---|---|---|
| (D-037) | fine-tuned MentalBERT | 0.9534 | 0.0008 | 0.9238 | not a D-040 arm |
| B | `all-mpnet-base-v2` | 0.7071 | 0.0104 | 0.5113 | **INDETERMINATE** |
| A | base MentalBERT | 0.5409 | 0.0084 | 0.2595 | **DEGENERATE** |

- **STRENGTHENED did not fire.** It required every non-fine-tuned arm at 0.85 or
  above. Neither arm was close.
- **NARROWED did not fire.** It required an arm below 0.70 *with HOC's assumptions
  intact*. Arm A is below 0.70 but fails the pre-registered non-singularity bar.
- **DISAGREE did not fire** on its pre-registered definition, which requires one arm
  at 0.85 or above alongside one below 0.70.

**Consequence for C1's scope: the claim does not narrow.** The NARROWED branch existed
precisely to catch the outcome "the fine-tuning-artefact explanation is sufficient, so
C1 applies only to fine-tuned representations." It did not fire, because removing the
fine-tuning does not produce a usable matrix. It produces a collapsing one. Chapter 1
therefore does **not** need the amendment D-040 provided for.

### The degeneracy gate earned its place, and this is on record

Read on its own, Arm A's 0.5409 looks like HOC recovering substantial real noise once
the suppression is removed, which is the NARROWED reading. `det(T)` = 0.2595 against
D-037's 0.9238 says otherwise: the rows are converging on one another. The gate was
written into D-040 **before the numbers existed**, for exactly this ambiguity, and it
is the only thing standing between this result and an incorrect amendment to Chapter 1.

This is the discipline D-034 records the absence of. Note the contrast honestly: in
D-034 the confound was articulated only after the result came back inconvenient, and
was logged as motivated reasoning. Here the distinction was drawn in advance and the
inconvenient reading is the one it blocked.

### Per-condition diagonals across all three representations

| true condition | fine-tuned | base | mpnet |
|---|---|---|---|
| depression | 0.9963 | 0.9577 | 0.9729 |
| eating_disorder | 0.9950 | 0.8265 | 0.9210 |
| schizophrenia | 0.9785 | 0.6448 | 0.8188 |
| bipolar | 0.9534 | 0.5409 | 0.7071 |

**Implied noise rate (1 minus the diagonal), with cleanlab from D-039 alongside:**

| condition | HOC fine-tuned | HOC mpnet | HOC base | cleanlab (D-039) | spread |
|---|---|---|---|---|---|
| bipolar | 4.6% | 29.3% | 45.9% | 15.2% | **10x** |
| schizophrenia | 2.2% | 18.1% | 35.5% | 10.1% | 16x |
| eating_disorder | 0.5% | 7.9% | 17.4% | 3.2% | 35x |
| depression | 0.4% | 2.7% | 4.2% | 2.1% | 11x |

**The rank order is identical in all four columns**: depression, then eating disorder,
then schizophrenia, then bipolar. Two independent estimators and three independent
representations agree completely on which conditions are noisier and by what ordering,
and disagree by an order of magnitude on how noisy any of them is.

**This is the sharpest form the C1 result has taken.** D-039 established that two
estimators disagree by a factor of three with no data-internal way to adjudicate.
D-041 adds that a *single* estimator disagrees with itself by a factor of ten
depending on which encoder it is handed, with each individual answer carrying a tight
cross-seed spread (0.0008 to 0.0104) that excludes the others. There is no
representation among the three at which HOC produces a matrix that can be trusted as
a point estimate, and nothing in the estimator's own diagnostics selects between them.

### The 2-NN coupling persists, and that is the finding

| arm | Pearson, HOC diagonal vs 2-NN agreement |
|---|---|
| fine-tuned | 0.9985 |
| base | 0.9940 |
| mpnet | 0.9952 |

D-037 measured this at 0.999 on the fine-tuned features and used it to argue that
HOC's output there was a deterministic function of a statistic D-036 had shown to be
largely manufactured by fine-tuning. **The open question was whether the coupling was
itself an artefact of that representation. It is not.** It holds at 0.994 or above in
three representations whose bipolar 2-NN agreement differs by roughly 60 percentage
points (fine-tuned 86.7%, base 26.9% within-E). HOC's output tracks the 2-NN consensus
structure of whatever space it is given, and that space is a free choice made by the
practitioner.

**Caveat that must travel with this number.** The correlation is computed across
K = 4 conditions, so it has two degrees of freedom, and a high value is easy to attain
whenever both quantities share a monotone ordering, which they do here by construction.
Taken alone it is weak evidence. Its weight comes from three things together: it is
reproduced at 0.994 or above in three representations, the mechanism predicts it a
priori (HOC fits its matrix to 2-NN consensus counts), and D-040 pre-registered the
above-0.9 threshold before the run. It also cannot distinguish "HOC's output is a
function of 2-NN agreement" from "both are functions of the same underlying
separability", but for the argument being made those are the same statement.

### Arm B's verdict is knife-edge, and this is stated before anyone finds it

Arm B's diagonal, 0.7071, sits **0.0071 above** the 0.70 narrow threshold. Its own
cross-seed standard deviation is **0.0104**. The margin is smaller than one standard
deviation. Its mean `|det|`, 0.5113, sits 0.0113 above the 0.5 non-singularity bar.

The branch stands as it fired, because the thresholds were fixed in D-040 before the
run and moving them now would void the entire exercise. But **INDETERMINATE versus
NARROWED for Arm B is not robust to a threshold shifted by one hundredth**, and the
thesis must say so wherever the branch is quoted. Nothing in this entry leans on Arm B's
branch label; the load-bearing content is the diagonal values and the coupling, both of
which are unaffected.

### Confound and limitations, carried forward from D-040

- **Neighbour starvation is identical across arms** and therefore does not invalidate
  the comparison, because `sample_size` was fixed at 15000 in every arm. It does bound
  what any arm can show: at |E| = 15000 a bipolar post has roughly 566 same-class
  candidates versus roughly 4221 in the full split (D-036).
- **Arm B does not test Theorem 5.5.** Liu et al.'s Definition 5.3 defines
  disentanglement as conditional independence of feature components **given the true
  label**, and true labels are unavailable here, so the property cannot be verified by
  any run on this data. Arm B tests the weaker observable proxy D-040 specified.
- **`all-mpnet-base-v2` is not purely self-supervised** in SimCLR's sense; it is
  trained on large-scale sentence-pair data, so the analogy to Liu et al.'s middle row
  is approximate.
- **The claim this supports:** the failure to identify a magnitude persists across
  representations that differ in how they were supervised. **The claim it does not
  support:** that a fully disentangled representation would not rescue the estimator.

### What this licenses for C2

The structure/magnitude split is the useful output. Four independent estimates agree on
the ordering and on bipolar being the noisiest condition, which is also the direction
the clinical literature predicts (D-018, open item 6). None of them pin the magnitude,
and their disagreement is not noise: each is internally tight and they are mutually
exclusive. That is the precise shape of the input Sesia's Theorem 4 needs and cannot
get from data: a **region** rather than a point. C2's elicitation supplies the
magnitude as a region while inheriting a direction the data already agrees on.

**Consequences.**

- C1 is complete on three legs: two estimators (D-037, D-039) and three
  representations (D-041). No further estimator or arm is needed to make the point.
- Open item 1 is closed for every arm: per-seed spread, prior, determinant and
  assumption flags are persisted per arm as `hoc_per_seed.csv`, and each arm's branch
  as `hoc_d040_verdict.csv`.
- D-040's prediction miss is added to the record. It is a pre-registered prediction
  that was partly wrong, which is the fourth such entry and the reason this log is
  worth having.

**Still to append to this entry** (measured, not yet transcribed): the off-diagonal
structure of each arm's mean T, specifically whether bipolar-to-depression remains the
dominant off-diagonal as the representation changes, which is the clinically predicted
direction and has been consistent across D-034, D-037 and D-039; each arm's estimated
prior p against the observed noisy proportions; each arm's per-condition 2-NN agreement
values; and the outcome of the `SentenceTransformer.encode()` parity check on Arm B's
pooling. None of these change any branch above.

**Links.** Reports the experiment pre-registered in D-040. Extends D-037's coupling
finding to two further representations. Extends D-039's confident-disagreement finding
from across-estimators to within-estimator-across-representations. Does **not** reverse
D-035. Feeds C2 (D-027, D-028).

---

# Part 6 · Writing

## D-013 · Follow the "Jazzify" thesis structure
**Date:** undated · **Status:** ACTIVE · **Category:** Writing

**Decision.** Structure follows the supervisor's own past thesis, cross-checked against three IIT/Westminster exemplars (Ammar W1761196, Hashim w1957407, AutoDistil-KG w1954098).

**Conventions confirmed.** Literature Review: Concept Map → Problem Domain → Technical Review → Datasets → Benchmarking → gap-pointing Summary, with the Existing Work table in an Appendix (Citation / Summary / Limitation / Contribution). Methodology: Saunders Research Onion table → Development Methodology → Project Management → Resources → Risks, with system architecture deferred to a separate Design chapter. Research Objectives table mapped to Learning Outcomes and Research Questions.

**Note.** Exemplars use a Layer/Choice/Justification table for the Research Onion with no diagram required. The diagram (Figure_3_1_Research_Onion.png) is optional supporting material.

---

## D-014 · Adopt supervisor's prose rules
**Date:** undated · **Status:** ACTIVE · **Category:** Writing

**Decision.** All chapters follow Sachindu Jayasinghe's stated rules.

1. **Do not ascribe intent or deliberateness to prior authors** when describing limitations. Avoid "deliberately," "by design," "by its authors' explicit design," "agnostic," "stop short." State plainly what a method does or does not do. (E.g. "does not model a noise distribution and is applied uniformly," not "deliberately noise-distribution-agnostic by design.") This applies even when the underlying factual contrast is correct.
2. **Do not open statements with combative negations** such as "No existing system…," "However, no identified work…". Lead with what the project does; let the "not yet combined" observation land plainly at the end.
3. **Do not praise the project's own honesty.** Remove "honestly," "remaining honest about," "we are transparent that," "made explicit rather than elided." Use neutral verbs: "acknowledging," "the analysis reports…"
4. **Unifying rule:** describe what prior work does and where your scope differs, without ascribing intent, without leading on a negation, without narrating your own virtue.
5. **UK English throughout** (characterise, modelling, labelling). Chapter 1 drifted into mixed US spellings; monitor in all chapters.
6. **No em dashes in deliverables.** Use colons, parentheses, semicolons or commas. En dashes retained in page/number ranges.

**Note.** Rule 2 independently required the rewrite that D-020 forced anyway. The false claim was also a badly-phrased one.

---

# Part 7 · Open items

| # | Item | Blocking | Owner |
|---|---|---|---|
| 1 | ~~Run HOC and `cleanlab` on the real data~~ **CLOSED 2026-08-09.** HOC on three representations (D-037, D-041) and cleanlab (D-039) all run; per-seed spread and prior persisted per arm | was D-030 | done |
| 2 | Measure 2-NN *noisy*-label agreement per condition in MentalBERT space (**amended 2026-07-24: a one-sided falsification test, not confirmatory; see expanded note below**) | The clusterability diagnostic | Soshan |
| 3 | Verify the `ρ = (Pᵀ)⁻¹ρ̃` derivation | D-028, D-031 | Soshan + supervisor |
| 4 | Verify the prevalence arithmetic in D-018 against Zhu et al.'s Theorem 2. **Counts half CLOSED 2026-08-14** (D-018 amendment: 146,400 raw vs 139,898 deduplicated, both correct, provenance recorded); the Theorem 2 formula check itself is still open | The strongest result in the project | Soshan |
| 5 | Decide prune vs no-prune | D-031 | Soshan + supervisor |
| 6 | Source every elicited range to a specific clinical study | The main attack on C2 | Soshan |
| 7 | Read Bortolotti et al. (arXiv:2501.18060) in full | D-029 | Soshan |
| 8 | Read Ding et al. (arXiv:2306.09335) in full | One leg of C3's convergence table is from a snippet | Soshan |
| 9 | Aich et al. access outcome | D-009 | External |
| 10 | Chapter 2 (Literature Review) draft | Next chapter | Soshan |
| 11 | Re-verify novelty before camera-ready | D-017; the noisy-conformal area published 4+ items in 2024–25 | Soshan |

---

## What the 2-NN clusterability statistic can and cannot show (open item 2, amended)

**Date:** 2026-07-24 · **Amends:** open item 2 · **Category:** C1

**Amendment.** The 2-NN *noisy*-label agreement statistic is confounded between the
noise level and clusterability failure and cannot distinguish them. It functions as
a **one-sided test**: high agreement would falsify the clusterability-failure
argument, whereas low agreement is consistent with it but does not establish it. The
**seed-instability of the noise estimators** (running HOC and cleanlab across seeds
and showing they disagree or wander) is the primary diagnostic for C1's failure
claim; the clusterability statistic is descriptive support, not the proof.

**Why this changes what open item 2 is for.** Item 2 was originally read as "measure
this and low bipolar agreement proves clusterability fails." That over-reads it. Two
different generative facts produce the same low agreement: (a) the feature geometry
is genuinely not clusterable for that condition, or (b) the proxy labels are simply
noisier for that condition, so neighbours in a perfectly clusterable space still
disagree on the *noisy* label. Measured against noisy labels, the statistic cannot
separate (a) from (b). So a low number is consistent with the argument but is not by
itself evidence for it; only a high number is decisive, and only in the falsifying
direction.

**How it is reported.** The statistic is reported per condition (not pooled) as
descriptive context, with the one-sided caveat and the noisy-vs-true-label caveat
stated in-line on every report. Both caveats are baked into
`src/noise/clusterability.py` (`NOISY_LABEL_CAVEAT`) so they cannot be dropped when
the numbers are copied into the thesis. Zhu, Song and Liu (2021, Table 3) report
roughly 78 to 88 percent feasible 2-NN tuples on noisy CIFAR-10 against **true**
labels; that is not comparable to a **noisy**-label agreement number.

**Consequences.** C1's load-bearing evidence shifts to estimator seed-instability and
cross-estimator disagreement (open item 1), with the arithmetic in D-018 (no
finite-sample rate for three of four conditions) as the theory. Clusterability
agreement is retained as a cheap, honest, one-sided check that can only ever *hurt*
the argument if it comes back high for a rare condition.

**Links.** Depends on D-030 (C1 is a diagnostic). Feeds open item 1.

---

## Elicitation sources still needed (open item 6, expanded)

The elicited set needs each range tied to a specific study. The *directions* are clinically predictable, which is the answer to "your bipolar threshold is just your own belief":

- **Bipolar to depression should dominate the off-diagonal**, for two independent documented reasons: **diagnostic delay** (bipolar is characteristically misdiagnosed as unipolar depression for years, so a truly-bipolar person may sincerely self-identify as depressed) and **phase predominance** (depressive phases occupy far more time-in-illness than manic ones).
- **Schizophrenia's diagonal should be lower**, because impaired insight is a core clinical feature, making self-identification less reliable than for mood disorders.
- **Depression's diagonal should be highest**: high base rate, high public awareness, lower stigma relative to psychosis.

**WARNING.** These three claims were stated from general knowledge during this session and are **exactly the kind of thing that sounds right and turns out subtly wrong.** Each needs a citation to a specific study. Where the literature is thin, **the range goes wide**, and the envelope honestly propagates that width into wider prediction sets. That propagation is a feature: vagueness becomes visible instead of hidden.

---

## Key references, verified by reading the full paper

| Reference | Role |
|---|---|
| Sesia, M., Wang, Y.X.R. and Tong, X. (2025) 'Adaptive conformal classification with noisy labels', *JRSSB* 87(3), 796–815, doi:10.1093/jrsssb/qkae114 | C2's foundation. Theorem 4 is the slot being filled. |
| Penso, C., Goldberger, J. and Fetaya, E. (2025) 'Conformal prediction of classifiers with many classes based on noisy labels', COPA, PMLR 266, 1–14 | The elicitable parameterisation (D-028). States that for a general noise matrix "all finite sample correction terms are not effective." |
| Zhu, Z., Song, Y. and Liu, Y. (2021) 'Clusterability as an alternative to anchor points when learning with noisy labels', ICML, PMLR 139, 12912–12923 | Killed D-024. Theorem 2 gives D-018's arithmetic. |
| Zhu, Z., Wang, J. and Liu, Y. (2022) 'Beyond images: label noise transition matrix estimation for tasks with lower-quality features', ICML, PMLR 162 | Rescued the project. Evidence for D-030. |
| Liu, Y., Cheng, H. and Zhang, K. (2023) 'Identifiability of label noise transition matrix', ICML, PMLR 202, 21475–21496 | Underpins D-018. Read in full 2026-08-09: it documents **four** identifiability routes, not three; the fourth (disentangled informative features, Theorem 5.5) prompted D-040. |

## Key references, cited but NOT yet read in full

| Reference | Risk |
|---|---|
| Ding et al. (2023, NeurIPS), 'Class-conditional conformal prediction with many classes', arXiv:2306.09335 | One leg of C3's convergence table. |
| Bortolotti et al. (2025), arXiv:2501.18060 | Bears on D-029. |
| Einbinder et al. (2024), *JMLR* 25(328), 1–66 | Context-setting only; lower risk. |

---

## Reversal scoreboard

| Claims made from abstracts or snippets | 4 |
| Of those, materially wrong | **4** |
| Claims made after reading the full paper | 6 |
| Of those, materially wrong | **0** |
| Pre-registered predictions recorded before a run | 4 |
| Of those, held in full | 2 (D-035's rule, D-036/D-037) |
| Of those, falsified or partly missed | 2 (D-034 falsified, D-040 partly missed) |

**This table is the argument for D-033.** Keep it updated. If it ever shows a wrong claim made after reading a full paper, that is worth knowing too.

The sixth full-paper claim is D-040's, from Liu, Cheng and Zhang (2023): that there is
a fourth identifiability route and that HOC's accuracy depends materially on the
feature extractor. D-041 bore it out, and more strongly than expected: the bipolar
diagonal moves from 0.5409 to 0.9534 across three encoders on identical data.

**The second block is the more useful one, and it is deliberately separate.** A project
that only recorded predictions it got right would be recording nothing. Two of four
pre-registered predictions did not hold as stated, both are written up in full
(D-034, D-041), and in both cases the pre-registration is what stopped the result being
reinterpreted after the fact.
