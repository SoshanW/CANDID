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
| D-011 | MentalBERT with unweighted cross-entropy as the baseline (corrected 2026-08-18) | ACTIVE | Engineering |
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
| D-042 | Sesia's Algorithm 2 interface pinned; degeneracy is closed-form, and it strengthens C1 | ACTIVE | C1/C2 |
| D-043 | Pre-registration for arm (c): closed-form brackets, judged predictions, and the head-to-head | OPEN (pre-registration, awaiting review and commit) | C2 |
| D-044 | Pretraining-corpus overlap between MentalBERT and the label-generating subreddits | ACTIVE | C1/Data |
| D-045 | MentalRoBERTa replication result: MentalBERT retained as the encoder of record | ACTIVE | Engineering |
| D-047 | Elicitation scoped against the budget: the binding quantity is label PPV, not misdiagnosis | ACTIVE | C2 |
| D-048 | Prediction function for all three arms: Sesia's plain threshold, equation (3) | ACTIVE | C2 |
| D-049 | Arm of record: the conservative Algorithm 2, with the optimistic variant reported alongside | ACTIVE | C2 |
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

**Current instances.** The elicitation-validity assumption (D-031); the Ding pooling tension; the Theorem 1 versus Theorem 2 distinction in the prevalence table; the DAIC-WoZ construct-validity problems; the closed-world limit of a 4×4 contamination model, which has no row for authors with none of the four conditions (added 2026-10-08, D-047).

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

## D-011 · MentalBERT with unweighted cross-entropy as the baseline
**Date:** undated (Milestone 0 complete) · **Status:** ACTIVE · **Category:** Engineering

**Decision.** MentalBERT (`mental/mental-bert-base-uncased`) fine-tuned for multi-class classification on the proxy labels with a plain HuggingFace `Trainer` and unweighted cross-entropy. 3 epochs, batch size 16, 256-token cap.

**Result.** 96% accuracy, 0.8782 macro-F1 on the held-out test set.

**Reasoning.** Unweighted cross-entropy is the right loss for this baseline for three reasons that hold independently of how the entry came to be written.

1. **C2 consumes the model's probabilities, and reweighting moves them.** Sesia's Algorithm 2 operates on the predicted class probabilities `π̂` (D-042). Inverse-frequency weighting shifts the fitted posterior away from the sampling distribution by design, so a reweighted head would hand the conformal step scores that no longer estimate `P(Y | X)` on the distribution the calibration set is drawn from. The unweighted head keeps the probabilities on the scale the downstream method expects, and any recalibration then happens once, in the open, in C2.
2. **D-003 makes the imbalance evidence rather than a nuisance.** The argument is that the conditions with the least data are the conditions whose noise is hardest to estimate. A loss that flattens the class ratio suppresses exactly the effect the thesis is about, and it would leave the baseline table looking better on bipolar for a reason unrelated to whether bipolar is learnable from this text.
3. **It keeps the C1 arms comparable.** D-035, D-036 and D-041 compare the geometry of the fine-tuned space against base MentalBERT and mpnet on identical data. A reweighted objective reshapes that geometry, and it would sit inside the fine-tuned arm as a confound with no counterpart in the other two.

The result also answers the concern the weighting was meant to address: at 0.8782 macro-F1 under a 20:1 ratio, the model is plainly not ignoring bipolar. A macro-averaged metric would have exposed it if it were.

**Imbalance is handled elsewhere, and not in the loss.** Four separate mechanisms carry it, each visible and each auditable on its own:

- **The split** is author-grouped rather than stratified (D-010), with per-condition proportions verified empirically to within 0.6 percentage points and roughly 490 bipolar examples in each of validation and test.
- **Reporting is per condition**, never pooled, throughout C0 and C1. Every diagnostic table in D-034 through D-041 is broken out by condition for this reason.
- **Macro-F1 is the headline metric**, so a rare condition counts as much as a common one in the number that gets quoted.
- **C2 targets label-conditional (Mondrian) coverage** rather than marginal coverage (D-029), which is where the per-condition guarantee is actually made rather than approximated by a loss term.

**Consequences.** The baseline loss knows nothing about *label noise*, which is what C2 exists to address. It also does nothing about imbalance, which is deliberate per the above.

---

**Correction (2026-08-18).** This entry previously read: *"Class-weighted MentalBERT fine-tune via a `WeightedLossTrainer` (HuggingFace `Trainer` subclass), inverse-frequency weighting"*, with the reasoning *"Plain cross-entropy under 20:1 imbalance would let the model ignore bipolar entirely and still look accurate."* That describes a class weighting scheme that was documented but never implemented. No `WeightedLossTrainer` exists in this repository and none ever did: `src/modeling/train.py` constructs a plain `transformers.Trainer`, and the D-039 cleanlab cross-validation folds construct the same one.

The 20:1 sentence is retained here as the rationale that was written down and then not carried into code. It is kept because it explains what an examiner will otherwise assume the baseline did, and because the Reasoning section above now takes the opposite position on its merits.

*How it was caught.* Repository inspection during the D-044 write-up, searching for the class name the entry asserts and finding no definition. The code did not fail and no run misbehaved, so nothing would have surfaced this on its own: an unweighted `Trainer` trains cleanly, and the only symptom was a log entry describing a component that was not there. The mismatch had already been flagged at two call sites, `src/noise/oos_probabilities.py:21` and `notebooks/train_colab.ipynb`, where the note records that the folds match the actual unweighted baseline. D-011 itself was never brought into line, so the authoritative log was the last place still carrying the error, which is the failure mode the log is supposed to prevent.

*What this does and does not change.* The D-039 folds are unaffected, since they reproduce whatever the baseline actually did and the baseline was unweighted throughout. What changes is the description and the justification: the reported macro-F1 comes from unweighted cross-entropy, and the reason for that is now recorded as a choice rather than left as an omission.

---

**Encoder choice: MentalBERT rather than MentalRoBERTa.**

Recorded here for the first time, having been an undocumented assumption until 2026-08-18. No benchmarked comparison between the two encoders was run on this project's data. The choice rests on precedent and on comparability with prior mental-health screening work, which predominantly reports MentalBERT.

Ji et al. (2022) release both variants, pretrained by continued domain-adaptive training on an identical corpus, so the comparison is close to a controlled one. Their own Tables 2 and 3 favour MentalRoBERTa on most of their evaluation datasets, including SWMH (F1 72.16 for MentalRoBERTa against 71.11 for MentalBERT). SWMH is the closest analogue to this project's task: multi-class, subreddit-derived, weakly labelled.

This is a known viva exposure with no current answer. An examiner asking "why the weaker of the two encoders the same paper released?" would be entitled to one, and precedent alone is a thin reply when the authors' own numbers point the other way.

**Commitment.** One MentalRoBERTa fine-tune, run under the identical protocol, splits and seed, reported as a **Milestone 0 baseline replication** and not as a C1 arm. The distinction matters: it is a robustness line under the baseline table, testing whether the C0 headline figure is encoder-specific. It does nothing for the corpus-overlap question in D-044, because MentalRoBERTa shares the identical pretraining corpus and therefore carries the identical exposure to the label-generating subreddits. Tracked as Part 7 open item 12.

**Links.** Amended by D-044 (the pretraining corpus behind this encoder choice). Result recorded in D-045. Feeds the C0 result table.

**Amendment (2026-09-27).** The commitment above has been carried out. Result and decision recorded in D-045: MentalRoBERTa scores 0.8841 macro-F1 against this entry's 0.8782 (Δ +0.0060, test-set bootstrap 95% CI [−0.0021, +0.0139]), and MentalBERT is retained as the encoder of record. Open item 12 is closed.

---

## D-045 · MentalRoBERTa replication result: MentalBERT retained as the encoder of record
**Date:** 2026-09-27 · **Status:** ACTIVE · **Category:** Engineering

**Decision.** MentalBERT was retained as the encoder for C0, C1 and C2, and the MentalRoBERTa fine-tune was recorded as a robustness line under the C0 baseline table, marginally ahead of MentalBERT and not separable from it on macro-F1.

**Knife-edge statement, first.** No numeric threshold separated "small" from "large" before the run. The only rule on record was qualitative, in notebook section 9b: *"a small gap supports the precedent argument; a large one is a limitation to state."* The classification below as "small" was therefore made after the number was seen. Two reference points existed before the run and both are used: Ji et al.'s own SWMH gap of 1.05 F1 points (D-011), and whether a test-set bootstrap interval excludes zero. This run is not counted as a pre-registered prediction on the scoreboard.

**Context.** D-011 recorded the MentalBERT choice as resting on precedent, with Ji et al. (2022) Tables 2 and 3 favouring MentalRoBERTa, and committed to one MentalRoBERTa fine-tune under the identical protocol (open item 12). The run used notebook section 9b, writing to a separate `Models_roberta/` root.

**Protocol integrity.** Only `--model-name` changed: 3 epochs, batch 16, LR 2e-5, 256 tokens, fp16, seed 42, unweighted cross-entropy, final-step checkpoint. Split fingerprints (SHA-256 of `hash_pandas_object`, first 16 hex) match on all three splits: train 111,892 rows `90162cd1b3d6678b`, val 13,967 `86ea56a8f2c55968`, test 14,039 `256e8754aa0700a8`. The prediction files are row-aligned.

**Evidence.** One seed per encoder; raw numbers in full in `Docs/roberta-replication-results.md`.

| Condition | Support | MentalBERT F1 | MentalRoBERTa F1 | ΔF1 | 95% CI (row bootstrap, 2,000) |
|---|---|---|---|---|---|
| depression | 11,351 | 0.9787 | 0.9806 | +0.0019 | [+0.0004, +0.0033] |
| eating_disorder | 1,421 | 0.9451 | 0.9557 | +0.0106 | [+0.0034, +0.0178] |
| schizophrenia | 786 | 0.8372 | 0.8368 | −0.0004 | [−0.0167, +0.0151] |
| bipolar | 481 | 0.7516 | 0.7635 | +0.0118 | [−0.0088, +0.0336] |
| **macro-F1** | 14,039 | **0.8782** | **0.8841** | **+0.0060** | [−0.0021, +0.0139]; author-clustered [−0.0015, +0.0137] |
| accuracy | 14,039 | 0.9605 | 0.9634 | +0.0029 | McNemar exact p = 0.0196 (120 BERT-only correct, 160 RoBERTa-only) |

Probability quality: NLL 0.1904 vs 0.1896; 15-bin ECE 0.0311 vs 0.0299. Minority-to-depression confusions: bipolar 116 vs 107, eating_disorder 79 vs 55, schizophrenia 133 vs 128. The two encoders agree on 97.72% of test predictions.

**Reasoning.**

1. **The gap is small on both pre-existing reference points.** +0.60 macro-F1 points is in the direction Ji et al. report and below their 1.05-point SWMH gap, and the macro-F1 bootstrap interval, row-level and author-clustered, includes zero.
2. **The direction is real on the pooled measure, and this is stated rather than hidden.** McNemar on per-row correctness gives p = 0.0196, and the depression and eating_disorder F1 intervals exclude zero. The pooled measure is 81% depression; the macro measure, which is the project's headline (D-003, D-011), is driven by the two rarest conditions and does not separate. The accurate statement is "MentalRoBERTa is marginally better; the headline gap is not resolved from zero by test-set resampling, and training-seed variance is unmeasured."
3. **Everything C1 and C2 depend on replicates.** The per-condition F1 ranking (depression > eating_disorder > schizophrenia > bipolar), depression as the dominant confusion sink, and the calibration of the softmax that Sesia's Algorithm 2 consumes (D-042) are all unchanged. The C1 finding is about structure versus magnitude across representations (D-041); a half-point encoder difference on the classifier does not bear on it.
4. **Switching would cost the comparability that C1 is built on.** D-035, D-036, D-038, D-039 and D-041 all use MentalBERT, fine-tuned or base. Moving C2 to MentalRoBERTa would place the abstention layer on a representation that none of the C1 diagnostics measured, in exchange for a gain the headline metric cannot distinguish from zero.

**Consequences.**

- The viva answer to "why the weaker encoder?" is now empirical: on this data, under the identical protocol, the encoder choice moves macro-F1 by 0.006, not separable from zero on the headline metric, with identical error structure. The answer concedes that MentalRoBERTa is nominally ahead on three of four conditions.
- The C0 baseline table gains a MentalRoBERTa row, reported with the caveats above. The MentalBERT figure of 0.8782 remains the baseline of record.
- No C1 arm is rerun on MentalRoBERTa. A MentalRoBERTa C1 arm would be a new decision with its own pre-registration.
- Nothing here addresses D-044: both encoders share the identical pretraining corpus.

**Limitations carried forward.** One training seed per encoder. Threshold for "small" set post hoc (stated above). Proxy labels, in-distribution Reddit test set only.

**Links.** Resolves the D-011 commitment and closes open item 12. Leaves D-044 unchanged. Feeds the C0 result table.

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

**Amendment (2026-10-26): a numerical check, not a verification.** Roadmap step 4's generator (`src/conformal/synthetic.py`) draws clean labels from `NOISY_PRIOR`, corrupts them through a known T independently of the scores, and recovers ρ with `v_region.transition_to_v`, the code every C2 number depends on. At n = 10⁶ and seed 11, three matrices (bipolar → depression 0.15 with background 0, 0.01 and 0.02 on the other off-diagonals; the last with bipolar → depression 0.30) recover bipolar at 0.0381, 0.0382 and 0.0380 against a true 0.038, while the observed noisy bipolar share is 0.0324, 0.0414 and 0.0443. Every component lies within four delta-method standard errors; the largest deviation, depression at z ≈ −2.5, is the realised clean draw itself (0.8109, z = −2.73 against the population prior), which the recovery tracks. Over 200 seeds at n = 2 × 10⁵ the recovery z-scores have mean −0.06 to 0.05 and standard deviation 1.01 to 1.05 per condition: unbiased, with a calibrated standard error. Pinned by `tests/test_conformal_synthetic.py`.

What this does and does not establish. It confirms that the code inverts ρ̃ = Tᵀρ correctly when T is known exactly and Assumption 1 holds by construction; that identity is the law of total probability, so the check guards the implementation and the orientation, not the novelty. It says nothing about recovery under a misspecified T, about the pruning observation, or about whether the derivation is already published. **Open item 3 stays open** pending the supervisor's check.

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

**Amendment (2026-08-18): DAPT exposure does not explain the base-space ordering.**
D-044 maps which of the four label subreddits appear in MentalBERT's pretraining
corpus, which raises the question of whether the base arm is quietly label-informed.
The ordering in the table above rules that reading out: the two conditions with no
corpus exposure bracket the sibling-exposed one rather than sitting on one side of
it, and all three rare conditions land at the same lift over chance. Exposure and
base-space separability are unrelated here. See D-044 for the exposure map and for
what it does imply.

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

## D-042 · Sesia's Algorithm 2 interface pinned; degeneracy is closed-form
**Date:** 2026-08-14 · **Status:** ACTIVE · **Category:** C1/C2

**This entry is a derivation, not a pre-registration.** No run produced it. It records
what reading Sesia, Wang and Tong (2025) in full establishes about what Algorithm 2
consumes, plus arithmetic evaluated on matrices already on record in D-037 and D-039.
The distinction matters for the scoreboard: this is the seventh claim made after
reading a full paper, and unlike the six before it, it has not yet been checked
against a run. The pre-registration for that run is deferred to D-043.

Working document: `Docs/c2-interface.md`. Code: `src/conformal/`, reproduced by
`python -m src.conformal`, with `tests/test_conformal_v_region.py` pinning the numbers
below.

---

### 1. What Algorithm 2 consumes

The noise input is a `1 - α_V` simultaneous confidence region on the **off-diagonal
entries of `V = M⁻¹`**, supplied as an interval box `[V̂_low, V̂_upp]`, together with
deterministic a priori bounds `V̄_upp` satisfying `max{|V̂_upp_kl|, |V_kl|} ≤ |V̄_upp_kl|`
almost surely for `l ≠ k`. It is not a set of matrices and not an ε-contamination ball.

Only the off-diagonals are required. M's rows sum to one, so `M·1 = 1`, hence `V·1 = 1`
and `V_kk = 1 - Σ_{l≠k} V_kl`.

### 2. C1's output is in the opposite direction, and the conversion is not free

Sesia's Proposition 1 defines `M_kl = P[Y = l | Ỹ = k]`. C1 estimates
`T_lk = P[Ỹ = k | Y = l]`. The conversion is:

```
ρ = (Tᵀ)⁻¹ ρ̃        M = diag(ρ̃)⁻¹ Tᵀ diag(ρ)        V = M⁻¹
```

`ρ̃` is observable. The clean prior `ρ` is recovered from `ρ̃ = Tᵀρ` rather than
assumed, so it requires no external input but inherits whatever error `T` carries and
can in principle leave the simplex. It does not for either matrix on record: HOC
fine-tuned gives 0.0369 / 0.8132 / 0.0961 / 0.0538 and cleanlab gives 0.0336 / 0.8199 /
0.0952 / 0.0513, against observed noisy proportions 0.038 / 0.812 / 0.096 / 0.054.

This is the concrete content of D-028. It also confirms the D-031 option (a) corollary
is real rather than notional: `α_V` enters the correction term (their equation 21) as
`2 α_V Σ_{l≠k}|V̄_upp_kl|`, so a deterministically containing region removes that term
outright.

### 3. The always-abstain failure mode is a closed-form inequality

Algorithm 2 constructs `Î_ci_k := {i ∈ [n_k] : i/n_k ≥ 1 - α - Δ̂_ci_k(S_(i)) +
δ_ci(n_k, n*)}` and sets `τ̂_k = 1` when that set is empty. By their Definition 1,
`τ_k = 1` places class k in every prediction set, which is the degenerate outcome
D-032 anticipated. Evaluating the membership test at `i = n_k`, where the class-k score
CDF equals one by construction, gives:

> **τ̂_k = 1 whenever δ_ci(n_k, n\*) > α + Δ̂_ci_k(S_(n_k))**, with
> `0 ≤ Δ̂_ci_k(S_(n_k)) ≤ Σ_{l≠k}|V̂_upp_kl| =: D̂_max`.

Testing `δ_ci > α` alone therefore over-flags degeneracy by at most `D̂_max`, and a
verdict is robust only when the margin exceeds it. Both quantities are reported below.

With `α_V = 0`, and noting that `|V̂_upp_kl| + δ̂^(V)_kl = |V̂_low_kl|` for negative
off-diagonals (which every matrix on record has):

```
δ_ci(n_k, n*) = c(n_k) + (2/√n*) · Σ_{l≠k}|V̂_low_kl| · min{K√(π/2), 1/√n* + √((log 2K + log n*)/2)}
```

Two structural consequences, each load-bearing:

1. **The binding quantity is the greatest total off-diagonal mass of V the region
   admits.** Region width and noise magnitude enter through the same channel, so they
   cannot be traded against one another.
2. **`n* = min_k n_k`.** The rarest condition's calibration count sets the correction
   term for every condition. Bipolar governs depression's threshold.

### 4. Evaluated on this project's numbers

**What is being evaluated, stated before the numbers.** Both regions below are built by
stacking C1's estimated matrices into a containment box, so they express **C1's
disagreement in V coordinates**. Neither is C2's input. C2's input is the elicited
region of D-028, which does not exist yet and is not evaluated here. What follows
therefore measures the *consequence* of C1's finding for a downstream consumer, which
strengthens C1; it does not test C2. Section 6 spells out why that distinction matters.

`α = 0.10`, `α_V = 0`, `K = 4`, calibration counts from the C0 test split (bipolar 481,
depression 11,351, eating disorder 1,421, schizophrenia 786), so `n* = 481`.

| region | contents | bipolar δ_ci | margin | D̂_max | verdict |
|---|---|---|---|---|---|
| A | D-037 and D-039 | 0.0919 | -0.0081 | 0.0804 | survives |
| B | all four C1 estimates | 0.2848 | +0.1848 | 0.0804 | degenerate, robust |

Per condition at region B: depression 0.0152 and eating disorder 0.0506 survive;
**schizophrenia reaches 0.1524, a margin of +0.0524 against a `D̂_max` of 0.0262, so it
degenerates robustly as well.** Two of four conditions are inadmissible at the C0
operating point.

Bipolar's region A margin of 0.008 does not survive `α = 0.05`.

**The calibration count cannot rescue region B.** `n*` enters at a `√n` rate, so
enlarging the calibration set moves δ_ci slowly: bipolar goes 0.2751, 0.2004, 0.1665,
0.1460, 0.1318 across calibration shares of 10, 20, 30, 40 and 50 percent of the
corpus. **Region B is inadmissible at α = 0.10 at every share.** It becomes admissible
only at α = 0.20 with a 30% share or more, which costs 80% label-conditional coverage
for a screening instrument *and* moves a fifth of the corpus out of training, degrading
the classifier and therefore the scores. That configuration is recorded as a measured
point on the frontier, not as a recommendation. **Region A is admissible at α = 0.10 at
every share, including the 10% already available.**

Feasibility frontier, the largest admissible `Σ_{l≠k}(|V̂_upp_kl| + δ̂^(V)_kl)` at
`α = 0.10`: bipolar 0.378, schizophrenia 0.411, eating disorder 0.441, depression
0.497. Region B's bipolar mass is **1.354**, which is 3.6 times the budget.

### 5. All four inputs are now measured, and a one-arm inference was wrong

D-041 transcribes only the **diagonals** of the base and mpnet arms. Both full matrices
were retrieved on 2026-08-14 from `Models/embeddings/train__mpnet/hoc_mean_T.csv` and
`Models/embeddings/train__base/hoc_mean_T.csv`. Neither directory holds a
`hoc_mean_T_rerun.csv`, so neither carries the ambiguity that `train/` does. Both
cross-check against D-041 (mpnet diagonal 0.7071, sd 0.0104, mean |det| 0.5113,
INDETERMINATE; base 0.5409, sd 0.0084, mean |det| 0.2595, DEGENERATE).

**A note on the base arm's invertibility, which Sesia's Theorem 4 requires.** Its
`hoc_d040_verdict.csv` records `nonsingular_all_seeds=False`. That field reports failure
of D-040's pre-registered |det| ≥ 0.5 quality bar, not singularity. Every seed in
`hoc_per_seed.csv` has `nonsingular=True` with |det| between 0.247 and 0.267, so the
matrix inverts and the conversion is valid. The field name invites the opposite reading
and is worth flagging wherever it is quoted.

**An inference drawn from one arm was wrong, and is corrected here.** While only mpnet
was measured, this entry recorded that the diagonal reconstruction understates bipolar's
off-diagonal mass in V (ratio 0.92) and concluded the degeneracy verdict was therefore
conservative. The base arm contradicts it: there the reconstruction **overstates**
bipolar's mass (ratio 1.19) and understates schizophrenia's (0.79). The error has no
stable sign across arms or across conditions, and a single arm never licensed the
generalisation. This is the same shape of mistake D-034 logs, caught earlier and at
lower cost, and it is the reason the reconstruction is now used only as a check and
never as an input.

**Both changes moved the result.** Bipolar's δ_ci fell from 0.3328 to 0.2848 and
remained robustly degenerate. Schizophrenia rose from 0.1251 to 0.1524 and moved from
**borderline to robust**, so the count of inadmissible conditions at the C0 operating
point is two, not one.

### 6. This strengthens C1. It does not test C2.

**The degeneracy result belongs to C1.** Region B is D-041's spread in different
coordinates, so its failure restates what D-041 already established. What passing it
through δ_ci adds is a *quantity*:

> D-041 shows the estimators disagree. It cannot say **how much** disagreement is too
> much, because that question has no meaning without a downstream consumer. Sesia
> supplies the consumer. Bipolar's admissible off-diagonal mass in V is 0.378 at
> α = 0.10, and region B sits at 1.354.

This closes an argumentative gap in C1. The question "the estimators disagree, so what,
perhaps it is close enough in practice" previously had no numeric answer, and now has
one: not close enough for the method that would consume it, by a factor of 3.6. C1
should carry this, and D-041 should be read alongside it.

**C2's claim is untouched.** C2 is not "use C1's spread as the region". It is that
clinical elicitation supplies a region the data cannot. No elicited region exists yet,
so nothing here evaluates it. What the arithmetic contributes to C2 is a **design
budget** that open item 6 previously lacked: the elicitation now has a numeric success
criterion (0.378 for bipolar at α = 0.10) rather than an open-ended reading task.

**Both outcomes remain reportable, as D-032 requires.** If the elicited region fits the
budget, C2 is a method and the contrast against 1.354 is the headline. If it does not,
the per-condition frontier is the characterisation result, in the sharper form of
"which conditions are admissible at which operating points" rather than "the coupling
does not pay off".

**Consequential changes.**

- The C2 proof-of-concept is re-planned (`progress-and-roadmap.md` §5). Steps 1 and 2
  are closed; the closed-form frontier replaces a simulated sweep as the main curve.
- α and the calibration fraction are reportable design variables, because both enter
  δ_ci directly. Neither is a route to feasibility for a region of region B's width.
- Open item 6 acquires a target, which changes how the elicitation should be scoped:
  wide-but-defensible ranges are only useful while they stay inside the budget, and
  D-031's decision not to prune interacts directly with that.

### 7. What is still open, and belongs in D-043

The closed form is sufficient for degeneracy and silent about everything short of it.
A run is still required for: mean set size for bipolar inside region A; whether
Assumptions 2, 3 and 5 hold empirically on this data; the D-032 head-to-head of arm (c)
against arm (b) at region A width; and the α at which each condition flips, measured
against the closed-form prediction above. That last item turns this derivation into a
falsifiable prediction, and D-043 should record it as one before the run.

### 8. Limitations named here rather than left for an examiner (D-019)

- **Assumption 1 (`Ỹ ⊥ X | Y`) is untested on this data and is not obviously true.**
  Which subreddit a person posts to plausibly depends on the text itself, not only on
  the underlying condition. Every guarantee in Theorems 2 to 5 rests on it.
- **The supplementary material was not available when this entry was written.**
  Sections S3, S4 and S5 carry the worked constructions of `V̂_low`, `V̂_upp` and
  `V̄_upp`, which are the template the elicited region should follow. Located
  subsequently: arXiv:2309.05092, 127 pages including all appendices.
- **Sesia's own route to V requires clean labels.** Their Section 3.3 and Proposition 4
  estimate `V = Q Q̃⁻¹` from a clean set `D₀` alongside a contaminated set `D₁`, with the
  region obtained by parametric multinomial bootstrap (Sison and Glaz, 1995). This
  project has no clean labels, which is what makes the elicited region the available
  route rather than a preferred one, and it means the head-to-head against their own
  estimation procedure cannot be run here.
- **The prediction function is not yet fixed.** Their equation (3) is the plain
  threshold on `π̂`; their Section 4 uses the generalised inverse quantile scores of
  Romano, Sesia et al. (2020). Set sizes differ between them and the choice must be
  recorded before the arms are compared.

**Links.** Implements D-027 and D-028. Confirms the D-031 option (a) corollary by
reading the equations. Consumes D-037, D-039 and D-041. Re-plans D-032's sweep.
Applies D-033. Pre-registration for the simulation deferred to D-043.

**Amendment (2026-10-26, D-049): the degeneracy frontier applies to the conservative
variant only.** Every δ_ci, budget and verdict in this entry is computed for Algorithm 2
with the set Î_ci_k of equation (22). Sesia et al.'s optimistic variant replaces it with
equation (25), whose `max{Δ̂_ci_k − δ_ci, −(1 − α)/n_k}` caps the threshold at standard
conformal's; at α = 0.10 its set Î_ci_k always contains i = n_k once n_k ≥ 9, so it never
sets τ̂_k = 1 on this criterion. Its coverage then rests on Proposition 3's condition, not
on Theorem 4. The numbers here are unchanged; their scope is narrowed to the variant
D-049 makes the arm of record.

**Amendment (2026-10-26): a derivation error in §3. The lower bound `0 ≤ Δ̂_ci_k(S_(n_k))` is false once the box has width.** Equation (20) (JRSSB p. 806; arXiv source label `eq:delta-hat-k-epsilon-bound`) is

> Δ̂^ci_k(t) := Σ_{l≠k} V̂^upp_kl (F̂^k_l(t) − F̂^k_k(t)) − δ̂^(V)_k* (K − 1) |F̂^k_k(t) − Σ_{l≠k} F̂^k_l(t) / (K − 1)| − |ζ̂^upp_k| Σ_{l≠k} |F̂^k_l(t) − F̂^k_k(t)|,

with δ̂^(V)_kl := V̂^upp_kl − V̂^low_kl (p. 805), δ̂^(V)_k* := max_{l≠k} δ̂^(V)_kl (equation (19), p. 806), and ζ̂^upp_k "a 1 − α_V upper confidence bound for" ζ̂_k := max_{l≠k}(V̂^upp_kl − V_kl) − min_{l≠k}(V̂^upp_kl − V_kl) (equation (19) and the text after it, p. 806). §3 bounded only the first term. The second and third are non-negative and are subtracted, so with any width Δ̂_ci can be negative. Then `δ_ci ≤ α` no longer keeps i = n_k in Î^ci_k, and the condition can enter every set. **Worked instance.** It is pinned by `tests/test_conformal_v_region.py::test_width_terms_can_degenerate_a_condition_that_passes_d042`.

- **Setup.** K = 4, α = 0.10, n_k = n* = 481. Every V̂^upp_kl = −0.02 and every off-diagonal interval has width 0.1, so δ̂^(V)_k* = 0.1, and ζ̂^upp_k = 0.1 as extracted from the box.
- **Correction term.** `v_region.correction_term` gives c(481) = 0.0283 and **δ_ci = 0.0965 < α**, so D-042's test calls the condition "ok".
- **Assumed CDFs.** The calibration points labelled k have distinct class-k scores, so F̂^k_k(S_(i)) = i/481. Every other label's class-k scores lie above S_(481), so F̂^k_l(S_(i)) = 0 for every l ≠ k and every i.
- **Equation (20).** The three terms are 0.06 i/n, 0.30 i/n and 0.30 i/n, so Δ̂_ci(S_(i)) = −0.54 i/n.
- **Equation (22).** It requires i/n ≥ 0.9965 + 0.54 i/n, that is 0.46 i/n ≥ 0.9965, which no i/n ≤ 1 satisfies. At i = 1, 240, 433 and 481 the right-hand side is 0.998, 1.266, 1.483 and 1.537. So Î^ci_k is empty and τ̂_k = 1, although δ_ci < α. The Algorithm 2 port returns τ̂_k = 1 on data realising these CDFs.
- **Zero-width comparison.** The same instance with a zero-width box gives δ_ci = 0.0397 and Δ̂_ci = +0.06 i/n. The first admissible index is i = 427, and the condition survives.

An independent check of equation (20) is recorded in D-043.

**Consequence.** Once the box has width, `δ_ci ≤ α` is **no longer sufficient** for a condition to survive.

- **The true necessary condition is `δ_ci ≤ α + D̂_max`.** As §3 already recorded, a borderline cell can survive.
- **Every robust degenerate verdict stands:** region B bipolar and schizophrenia here, and L1 and R0 bipolar in D-047. The upper bound `Δ̂_ci ≤ D̂_max` is sound, because both width terms only lower Δ̂_ci.
- **Every "survives" or "ok" verdict is unproven** pending D-043's run, in §4 and §6 here and in `c2-interface.md` §6 and §7. That includes region A bipolar's "survives by 0.008". The one exception is depression in region A, whose margin exceeds the largest possible width penalty.
- **The budget table (0.378 and the rest) is a necessary condition only,** up to the borderline margin §3 already named. A region over budget does not survive, except possibly at a borderline margin. A region within budget is not thereby shown to survive.

Counted on the reversal scoreboard as the first materially wrong claim made after reading a full paper.

---

## D-044 · Pretraining-corpus overlap between MentalBERT and the label-generating subreddits
**Date:** 2026-08-18 · **Status:** ACTIVE · **Category:** C1/Data

**Decision.** The overlap between MentalBERT's pretraining corpus and the subreddits that generate this project's proxy labels is recorded as a disclosed limitation, and the existing three-arm evidence is recorded as bounding its size.

**Context.** MentalBERT is BERT-base-uncased continued with masked language modelling (MLM only; the paper does not use next sentence prediction) on posts crawled from r/depression, r/SuicideWatch, r/Anxiety, r/offmychest, r/bipolar, r/mentalillness and r/mentalhealth, totalling 13,671,785 sentences (Ji et al., 2022, section 2.2). The proxy label in this project is subreddit membership, so the pretraining corpus is drawn from the same kind of source as the labels themselves.

**Per-condition exposure map.**

| Condition | Label subreddit (Low et al.) | Exposure in MentalBERT's DAPT corpus |
|---|---|---|
| depression | r/depression | exact match with the DAPT corpus |
| bipolar | r/bipolarreddit | sibling community; r/bipolar is in the corpus, r/bipolarreddit is not |
| schizophrenia | r/schizophrenia | not in the corpus |
| eating_disorder | r/EDAnonymous | not in the corpus |

**What the paper does and does not report.** It reports the subreddit list and the sentence count. It reports no collection dates, no post count, no deduplication procedure and no filtering procedure. Overlap between the DAPT corpus and this project's evaluation data therefore cannot be excluded or confirmed from the primary source. The correct statement is **"cannot be established either way"**, not "likely" and not "near-certain". Anything stronger is a claim the paper does not support.

**Reasoning: why this does not undermine D-035, D-036 or D-041.**

- If DAPT exposure had made base MentalBERT label-informed, the base arm should sit closer to the fine-tuned arm than a label-naive encoder does. D-041 measures the opposite. Base MentalBERT's bipolar diagonal is 0.5409, against mpnet's 0.7071 and fine-tuned 0.9534. Base is the furthest of the three from the fine-tuned geometry, and the label-naive encoder is the one in between.
- Exposure does not predict base-space separability either. D-036 base agreement runs 94.5% (exposed), 71.1% (unexposed), 41.2% (unexposed), 29.3% (sibling). On lift over chance the three rare conditions are indistinguishable at 7.45x, 7.53x and 7.77x, and depression's 1.16x is a ceiling effect at 81.2% base chance, as D-036 already records.
- Consequence: base MentalBERT remains usable as a control, and the control is conservative rather than compromised. Whatever exposure it carries pushes the measurement in the direction that would weaken the argument, and the argument survives anyway.

**Consequences.**

- The Milestone 0 figure in D-011 carries an additional reason to be discounted, and the reason is unevenly distributed across conditions: text contamination is possible for depression, weaker for bipolar, and absent for schizophrenia and eating disorder. Uneven contamination is worse for per-condition metrics than uniform contamination would be, because it inflates one condition's apparent performance relative to the others rather than shifting the whole table.
- C3 gains a limitation to state. Depression is the only condition with a satisfiable finite-sample guarantee under D-018, and it is also the only condition with exact-subreddit DAPT exposure. The two facts are independent in derivation: D-018's split comes from prevalence arithmetic on Zhu, Song and Liu's Theorem 2, not from anything about the encoder. The coincidence should be named in the thesis rather than left for a reader to find and read as circularity.

**Evidence.** Ji et al. (2022), sections 2.1 and 2.2, Tables 2 and 3; Low et al. (2020) subreddit list; the D-036 base-agreement table and the D-041 per-condition diagonal table, as cited above.

**Links.** Amends D-011. Bounds a confound in D-035 and D-036. Feeds the C3 limits analysis alongside D-018 and the neighbour-starvation finding in D-036.

---

## D-047 · Elicitation scoped against the budget: the binding quantity is label PPV, not misdiagnosis
**Date:** 2026-10-08 · **Status:** ACTIVE · **Category:** C2

**This entry is a derivation, not a run**, in the same sense as D-042. Working document:
`Docs/c2-elicitation-scoping.md`. Code: `src/conformal/elicited_region.py`, reproduced by
`python -m src.conformal.elicited_region`, with `tests/test_conformal_elicited_region.py`
pinning the numbers below.

**Decision.** Open item 6 is re-scoped. The elicitation's success criterion is a floor on
the worst-case **positive predictive value (PPV) of each rare label**: the share of posts
in r/bipolarreddit, r/schizophrenia and r/EDAnonymous written by someone who truly has
that condition. For bipolar at α = 0.10 and n\* = 481 the floor is about 0.78. The three
directional claims originally listed under open item 6 are kept, as sources for wide
ranges on entries the budget barely sees.

**Context.** D-042 supplied the budget (bipolar 0.378). Roadmap step 3b asked whether a
defensible reading of the clinical literature yields a region that fits under it. An
elicited region is a set of transition matrices, so it was modelled as independent
intervals on the off-diagonals of `T`, propagated to a box on V by vertex enumeration
(deterministic containment, α_V = 0), with an interior check that no sampled point
escaped.

**What the arithmetic shows.**

1. **Row k of V is built from the posts carrying label k.** Truly bipolar posts landing
   in r/depression leave the bipolar row of M at exactly (1, 0, 0, 0); the budget does
   not see the direction diagnostic delay predicts. Inflow into a rare label from
   depression is amplified by about ρ_depression / ρ̃_bipolar ≈ 21. As single entries
   with no width, depression → bipolar exhausts bipolar's budget at **1.3%**, depression
   → schizophrenia at 1.9% and depression → eating disorder at 3.5%, while bipolar →
   depression breaks only depression's budget, at 87.65%.
2. **Jointly, the cheap entries stop being cheap.** Region R0 takes the three directions
   wide (leakage into r/depression up to 60% and 40%), allows rare-to-rare inflow up to
   10%, and has **no** depression-to-rare inflow. Bipolar still degenerates robustly
   (mass 1.147, δ_ci 0.2457) and schizophrenia borderline. Leakage into depression
   raises the recovered clean prevalence of the rare conditions, which multiplies every
   rare-to-rare inflow.
3. **The frontier.** F1 (leakage ≤ 40%, rare-to-rare and depression-to-rare ≤ 0.5%)
   passes all four, bipolar mass 0.192, worst-case bipolar PPV 0.873. F3 (as F1 but
   depression → bipolar ≤ 1%) degenerates borderline, 0.392, PPV 0.763. L1 (PPV 0.45)
   degenerates robustly at 1.492. Across a grid of region shapes the lowest worst-case
   bipolar PPV that passed was 0.744 and the highest that failed 0.780.
4. The deduplicated prior from the D-018 amendment moves nothing material (0.8% passes,
   1% fails borderline).

**What the literature shows** (read status in the working document).

- Zimmerman, Ruggero, Chelminski and Young (2010): of 700 outpatients, 145 reported a
  prior bipolar diagnosis and 82 (56.6%) were not confirmed by the SCID, 68 of whom had
  lifetime major depression. The label's PPV in that clinic is 63/145 = **0.43**
  (derived, not stated).
- Birnbaum et al. (2017): 146 of 671 Twitter schizophrenia self-disclosers appraised
  genuine by clinicians (0.22; 0.37 counting "maybe"). Ernala et al. (2019): 153 of 635.
- **No study located measures the clinical validity of subreddit membership.** The
  clinic rate cannot transfer as a Reddit rate: 11% depression → bipolar would exceed the
  whole observed bipolar share, and the recovered prevalence leaves the simplex.
- Most documented contamination comes from people with none of the four conditions
  (jokes and quotes; borderline personality disorder and PTSD in the overdiagnosed
  group). A 4×4 model cannot represent it.

**Consequences.**

- **Projected at the C0 operating point:** bipolar inadmissible under any region the
  located literature supports; depression and eating disorder admissible in every region
  tested; schizophrenia open, pending a PPV the literature does not pin. This is D-032's
  characterisation branch in per-condition form, unless a structured elicitation can
  defend a bipolar PPV floor near 0.8.
- **A literature-leaning elicited region and C1's spread fail by similar factors**: L1 at
  1.49 against region B at 1.354, for different reasons. That parallel belongs in the
  thesis.
- **D-031 does not decide the bipolar verdict.** Pruning removes only the part of a region
  beyond the simplex ceiling (about 4.7% depression → bipolar); the budget is exhausted
  near 1%, so no region between those points is rescued by pruning.
- **A new decision is needed** (open item 14): whether to elicit PPVs directly. That is
  Sesia's direction, which D-028 set aside because a PPV depends on prevalence. For a PPV
  asked about these specific communities that objection weakens, but the change of
  direction must be recorded as a decision if taken.
- The closed-world limit joins Assumption 1 under D-019.
- D-043's pre-registration should carry these per-condition predictions so that the
  simulation can confirm or move them.

**Links.** Consumes D-042. Re-scopes open item 6. Bears on D-028, D-031, D-032 and D-043.
Applies D-019 and D-033.

**Amendment (2026-10-26, D-049): the frontier applies to the conservative variant
only.** The budget, the PPV floor of about 0.78 and every region verdict above inherit
D-042's δ_ci, which is that of Algorithm 2 with equation (22). Under the optimistic
variant, equation (25), no condition degenerates on this criterion, and the question
becomes whether Proposition 3's condition holds, which needs the true V. The numbers are
unchanged; their scope is the arm of record set by D-049.

**Amendment (2026-10-26): the frontier inherits D-042's derivation error.** Every verdict above uses D-042's δ_ci comparison, whose lower bound `0 ≤ Δ̂_ci` fails once the box has width (D-042 amendment of this date; equation (20)). Every region here has width. Therefore:

- **The robust degenerate verdicts stand:** L1 and R0 bipolar.
- **Every "ok" verdict is unproven pending D-043's run.** This covers all four conditions in F1 and F2, and depression, eating disorder and schizophrenia in F3, L1 and R0. The borderline verdicts (F3 bipolar, R0 schizophrenia) are also undetermined.
- **The budget, the PPV floor of about 0.78, and the grid boundaries are necessary conditions only,** up to the borderline margin D-042 already named. A region failing them fails D-042's test. A region meeting them is not thereby shown to survive.

The literature comparison is unaffected, since it already sits below the necessary floor.

---

## D-048 · Prediction function for all three arms: Sesia's plain threshold, equation (3)
**Date:** 2026-10-26 · **Status:** ACTIVE · **Category:** C2

**Decision.** Every arm of D-032 uses the prediction function of Sesia et al.'s equation (3), `C(x, τ) = {k : π̂(x, k) ≥ 1 − τ_k}`, with non-conformity score `ŝ(x, k) = 1 − π̂(x, k)` (equation (2)), deterministic, with empty sets allowed and reported.

**Context.** Roadmap step 5 and `c2-interface.md` §8 require the prediction function to be fixed before the arms are compared, because set sizes depend on it. Sesia, Wang and Tong (2025) state their results for any C satisfying Definition 1. Equation (3) is their worked example and the one named on the input line of Algorithms 1 and 2 ("Prediction function C satisfying Definition 1; e.g. (3)"). Their §4.1 simulations instead use the generalised inverse quantile scores of Romano, Sesia et al. (2020), and so does their public implementation: `cln/classification.py` in github.com/msesia/conformal-label-noise (Python, MIT licence, last pushed 2024-06-19) computes every score through `arc.classification.ProbabilityAccumulator`, randomised by one uniform draw per point (`random_noise`, passed to both `calibrate_scores` and `predict_sets`).

**Reasoning.**

1. **It makes the C2 object a per-condition threshold.** Under Mondrian calibration (D-029) and equation (3), condition k is flagged exactly when `π̂(x, k) ≥ 1 − τ̂_k`: one cut per condition on that condition's own probability. That is the "per-condition abstention threshold" of roadmap §7 in its plainest form, and it can be stated to a clinician in one sentence. For the generalised inverse quantile score, whether k enters is believed to depend on how k ranks against the other classes' probabilities. **That characterisation is unverified:** Romano, Sesia et al. (2020) and Sesia et al.'s Supplementary Section S1, which reviews the score, have not been read (D-033).
2. **It is deterministic.** Equation (3) involves no random draw, so the same post always receives the same set, a property a screening instrument should have. This is a property of equation (3), not a finding about the alternative: the reference implementation happens to randomise its scores, but whether a non-randomised variant of the generalised inverse quantile score exists is a question about Romano, Sesia et al. (2020), which is unread.
3. **It can be ported and checked line by line.** `src/conformal/` is numpy-only. `cln` imports `arc` (vendored under `third_party/`) and `statsmodels`, and its README lists torch. With equation (3) the score is one line, so Algorithms 1 and 2 can be ported from equations (12) to (22) and compared term by term against `cln`'s `compute_Delta_hat`, `compute_Delta_hat_bounded`, `compute_delta_const` and `compute_delta_const_bounded`, without taking the dependency.
4. **It leaves D-042 and D-047 untouched.** The degeneracy criterion uses only `F̂^k_k(S_(n_k)) = 1` and the Definition 1 rule that `τ_k = 1` puts k in every set. Both hold for any valid score, so no number in either entry moves.

**What it costs.**

- **Set sizes are not comparable with Sesia et al.'s Figures 1 and 2**, which use the other score. The POC compares arms with each other, not with the paper, so this is a reporting caveat, not a validity one.
- **Empty sets can occur.** Under equation (3), a post whose probabilities clear no condition's threshold receives the empty set. Definition 1 does not forbid it. An empty set counts as non-coverage, and the empty-set rate is reported per arm, because in screening it reads as "none of the four conditions", which the closed-world caveat (D-047) already marks as a case the model cannot represent.
- **If the other score gives materially smaller sets, every arm's sizes are larger than they need be.** Whether it does on this problem is unknown and the claim about its adaptivity is unverified. The arms take the score matrix as input, so a second-score robustness arm can be added later without touching the Algorithm 2 code.

**Evidence.** Sesia et al. (2025), §2 equations (2) to (5), Definition 1, the Algorithm 1 and 2 boxes, and §4.1, read in the local copy on 2026-10-26. `cln/classification.py` read in full on the same date. Romano, Sesia et al. (2020) not read; the reasoning does not depend on any claim about it beyond the fact, visible in Sesia et al.'s §4.1 and in `cln`, that it is the score they used.

**Consequences.** Arm (a) calibrates by equation (4): `τ̂_k` is the ⌈(1 + n_k)(1 − α)⌉-th smallest of `1 − π̂(X_i, k)` over calibration points with `Ỹ_i = k`, and equals one when that rank exceeds n_k. Arms (b) and (c) apply Algorithm 2 with the same scores. Reported per arm and per condition: coverage against the clean label, mean set size, empty-set rate and full-set rate.

**Links.** Implements roadmap step 5's open choice. Depends on D-029 and D-032. Neutral for D-042 and D-047. Precedes D-043.

---

## D-049 · Arm of record: the conservative Algorithm 2, with the optimistic variant reported alongside
**Date:** 2026-10-26 · **Status:** ACTIVE · **Category:** C2

**Decision.** Arms (b) and (c) report the conservative Algorithm 2 of Sesia et al. (set Î_ci_k of equation (22), guarantee by Theorem 4) as the arm of record. The optimistic variant (equation (18) for Algorithm 1, equation (25) for Algorithm 2; their "Adaptive+", guarantee by Propositions 2 and 3) is reported alongside it in every table. The roadmap's step 4 sanity gate is corrected to two tests.

**Context.** Reading equations (14) to (25) to fix D-048 showed two things the roadmap and D-042 had not separated. First, the optimistic set `{i : i/n_k ≥ 1 − α − max{Δ̂_ci_k(S_(i)) − δ_ci, −(1 − α)/n_k}}` always contains i = n_k once n_k ≥ (1 − α)/α (9 at α = 0.10), so it never sets τ̂_k = 1, and its threshold is never above standard conformal's. D-042's degeneracy frontier is therefore a property of the conservative variant only. Second, with T = I the noise terms vanish but δ keeps c(n_k), so the conservative variant does not coincide with standard conformal.

**Reasoning.** Proposition 3's condition, `inf_t Δ_k(t) ≥ δ_ci(n_k, n*) − (1 − α)/n_k`, involves the true coverage inflation factor `Δ_k(t) = F^k_k(t) − F̃^k_k(t)` (equation (7)), which needs the true V and the clean-label score distributions. C1 established that V is not recoverable from this data (D-041). On real data, therefore, only the conservative variant has a guarantee that the elicited region can support: Theorem 4 needs only that the region contain V with probability 1 − α_V, which D-031's deterministic containment supplies with α_V = 0. The optimistic variant is still reported, because Sesia et al. find it "preferable in practice" (§4.1) and because the synthetic harness, where V is known, can measure what the unverifiable condition costs.

**What the condition says, read from the proof.** As stated, the infimum is over all t ∈ ℝ. Both CDFs in Δ_k vanish below the range of the scores, so `inf_t Δ_k(t) ≤ 0`, and the condition can hold only if `δ_ci ≤ (1 − α)/n_k`. Since `δ_ci ≥ c(n_k)`, which is 0.0283 against (1 − α)/n_k = 0.0019 at bipolar's n_k = 481, the condition as written holds at no calibration count in this project. The proof of Proposition 2 (arXiv:2309.05092 v2, Supplement §A7; Proposition 3's proof builds on it) uses the hypothesis only through `|max{Δ̂_k(t) − δ + (1 − α)/n_k, 0} − (Δ_k(t) − δ + (1 − α)/n_k)| ≤ |Δ̂_k(t) − Δ_k(t)|` evaluated at the realised threshold `t = S^k_(î_k)`. The harness therefore checks both: the condition as stated, and the at-threshold form the proof uses. The second is this project's reading of the proof, not a statement Sesia et al. make. The JRSSB supplement was not checked, and its numbering may differ from the arXiv version.

**Gate correction.** Roadmap step 4 read "with T = I, all three arms must coincide". At T = I the conservative variant uses rank ⌈n_k(1 − α + c(n_k))⌉ against standard conformal's ⌈(n_k + 1)(1 − α)⌉:

| condition | n_k (C0 test split) | c(n_k) | standard rank | conservative rank |
|---|---|---|---|---|
| bipolar | 481 | 0.0283 | 434 | 447 |
| schizophrenia | 786 | 0.0222 | 709 | 725 |
| eating disorder | 1,421 | 0.0165 | 1,280 | 1,303 |
| depression | 11,351 | 0.0059 | 10,217 | 10,283 |

The gate is now two tests at T = I: the optimistic variant equals arm (a) exactly, and the conservative variant differs from it by exactly this shift. The original text was an internal inconsistency with D-042's own δ_ci, which contains c(n_k), not a misread source, so it is not counted on the reversal scoreboard.

**Arm (b) needs no new definition.** With a zero-width box (V̂_low = V̂_upp = V̂, α_V = 0), δ̂^(V)_k* = 0, and ζ̂^upp_k = 0 because the box asserts V = V̂. Equation (20) then reduces to equation (13) with V̂, and equation (21) to equation (17). Algorithm 2 with a point estimate is Algorithm 1 with V̂, the plug-in that Sesia et al.'s §3.3 describes, so D-032's wording and the roadmap agree.

**Evidence.** Sesia et al. (2025), equations (7), (13) to (22) and (25), Propositions 2 and 3, read in the local copy on 2026-10-26; proof of Proposition 2 in arXiv:2309.05092 v2, Supplement §A7, read the same day. c(n_k) from `v_region.expected_sup_deviation`. The reference implementation (`cln/classification.py`) codes the optimistic set as `max(−(1−α)/n_k, Δ̂ − δ)`, matching equation (18).

**Consequences.** Every arm table carries a conservative and an optimistic line. D-042 and D-047 carry dated amendments narrowing their frontier to the conservative variant; their numbers do not change. On the synthetic harness, step 6 records, per condition and run, whether Proposition 3's condition holds in both forms and whether the optimistic variant under-covers. **D-043 must pre-register a prediction for this**; none is made here.

**Links.** Depends on D-031, D-041, D-042 and D-048. Amends D-042 and D-047. Corrects roadmap step 4. Precedes D-043.

---

## D-043 · Pre-registration for arm (c): closed-form brackets, judged predictions, and the head-to-head
**Date:** 2026-10-26 · **Status:** OPEN (pre-registration; becomes ACTIVE when committed, before any arm (c) run) · **Category:** C2

**Decision.** The run specification, predictions, thresholds and branch rules below are fixed before arm (c) is run. Predictions come in two tiers, and only the first makes pass or fail calls on its own:

- **Primary.** What the closed form guarantees. For each cell, Δ̂_ci at the threshold lies between its lowest possible value and D̂_max, and each cell is classed **certain-ok**, **certain-degenerate** or **undetermined**. A certain cell that comes out otherwise is a bug, not a finding.
- **Secondary, a judgement.** Point predictions for the undetermined cells, using two stated heuristic factors. These are scored, but no pass or fail call about D-042, D-047 or the code rests on them.

Every number is arithmetic on the region boxes, computed with `v_region.correction_term`, `v_region.region_from_matrices` and `elicited_region.v_box`; none comes from a run of arm (c). The Algorithm 2 code arm (c) will use (`src/conformal/adaptive.py`, commit 3f54b41) has so far been run only with zero-width boxes (step 6) and in two unit tests on artificial boxes: one checks only the optimistic variant's selected index, the other uses an off-diagonal mass of 9 per row. No region in this entry has been run.

### Context

D-042's "survives" verdicts, and every "ok" verdict in D-047, rested on `0 ≤ Δ̂_ci`. Equation (20) subtracts two width terms, so that lower bound fails once the box has width. This is recorded as a derivation error in the D-042 amendment of this date, which quotes equation (20) and the definitions of δ̂^(V)_k* and ζ̂^upp_k (JRSSB pp. 805 and 806) and gives a worked instance. Two checks confirmed it. A subagent working from the arXiv LaTeX source answered both questions the same way: Δ̂_ci can be negative, and the "ok" verdicts do not follow, while the robust degenerate ones do. Its report notes that it had read access to DECISIONS.md, which then held an earlier draft of this entry. Its derivation from the equations is its own, but it was not blind to this draft. The same error is now on the reversal scoreboard.

**The closed-form bracket.** At t = S_(n_k), where F̂^k_k = 1:

- term one of equation (20) is at most D̂_max = Σ_{l≠k} |V̂^upp_kl| and at least −P_k, where P_k = Σ_{l≠k} max(V̂^upp_kl, 0);
- the second term subtracts at most (K − 1) δ̂^(V)_k*;
- the third subtracts at most (K − 1) |ζ̂^upp_k|.

So **Δ̂_min = −P_k − (K − 1)(δ̂^(V)_k* + |ζ̂^upp_k|) ≤ Δ̂_ci(S_(n_k)) ≤ D̂_max**, and the upper bound holds at every t. This gives:

- **certain-degenerate** when `δ_ci − α > D̂_max`: no i satisfies equation (22);
- **certain-ok** when `α − δ_ci ≥ −Δ̂_min`: i = n_k always satisfies it;
- **undetermined** otherwise.

**ζ̂^upp_k.** The paper defines ζ̂_k in equation (19) and ζ̂^upp_k as "a 1 − α_V upper confidence bound for ζ̂_k", which "can generally be extracted directly from [V̂^low, V̂^upp]" (p. 806). With α_V = 0 and V inside the box, every V̂^upp_kl − V_kl lies in [0, δ̂^(V)_kl], so ζ̂_k ≤ δ̂^(V)_k*. **Primary: ζ̂^upp_k = δ̂^(V)_k*,** the bound the box supports and the default in `adaptive.py`. **Sensitivity: ζ̂^upp_k = 0,** which the paper attains when the band is uniform (its randomised-response example). None of these boxes is uniform, so ζ̂ = 0 is not justified here. It is reported to show how much of the outcome rests on the third term.

### 1. Run specification (fixed)

- **Regions.** A and B from `region_from_matrices(REGION_A_MEASURED / REGION_B_MEASURED, NOISY_PRIOR)` (`c2-interface.md` §6). F1, F2, F3, L1 and R0 from `v_box(SCOPING_REGIONS[name], NOISY_PRIOR)` (D-047). α_V = 0, `V̄_upp` unused, ζ̂^upp_k = δ̂^(V)_k*. The whole run is repeated with ζ̂^upp_k = 0 and reported as a sensitivity; only the primary setting is scored.
- **Synthetic truth per region**, chosen inside the region so that Theorem 4's premise holds. A and B use `T_HOC_FINE_TUNED`; F1 to R0 use the midpoint of the T-interval box. The generator's clean prior is the recovered `ρ = (Tᵀ)⁻¹ NOISY_PRIOR`, so the population proxy frequencies equal `NOISY_PRIOR`, the prior the boxes were built with. Checked now: every recovered prior is on the simplex and every true V lies inside its box.
- **Harness as in steps 5 and 6.** Calibration counts 481 / 11,351 / 1,421 / 786; 100 repetitions; 20,000 test points; separability 2.0; base seed 0; prediction function D-048; conservative variant of record with the optimistic variant alongside (D-049). A 2 × 10⁶-point reference sample per region (seed base + 10⁶) for Δ_k and the assumption checks.
- **Outcome definitions.** Condition k is *degenerate* in a run when τ̂_k = 1. A cell's outcome is **degenerate** if τ̂_k = 1 in at least 90 of 100 runs, **admissible** if τ̂_k < 1 in at least 90, and **split** otherwise. *Valid* means coverage against the clean label ≥ 0.90 − 3 SE.

### 2. Prediction 1: admissibility at α = 0.10

**Primary (closed form).** Under ζ̂ = δ̂*: 4 cells certain-degenerate (B bipolar, B schizophrenia, L1 bipolar, R0 bipolar), 1 certain-ok (A depression), 23 undetermined. Under ζ̂ = 0: B depression also becomes certain-ok, leaving 22 undetermined. The certain cells must hold in 100 of 100 runs. Any exception is an implementation error: stop, fix and re-run, and do not interpret.

**Secondary (judgement).** For undetermined cells, predicted degenerate when `α − δ_ci + 0.5 D̂_max − P_k < 0.7 (K − 1)(δ̂* + ζ̂^upp)`. The factors encode an expectation about the synthetic scorer (separability 2.0): at the relevant thresholds, `|F̂^k_k − F̂^k_l|` should sit near 0.7, and term one near half its maximum. Confidence is **high** when the call is unchanged across factors in [0.4, 1.0] and term one anywhere in [0, D̂_max], and **low** otherwise. A-eating disorder is marked low because it clears that check by 0.0004. Under ζ̂ = 0 the only secondary call that changes is A-eating disorder, from degenerate to admissible.

| region | condition | δ_ci | D̂_max | Δ̂_min, ζ̂ = δ̂* | **primary class** | Δ̂_min, ζ̂ = 0 | class, ζ̂ = 0 | D-042 / D-047 | secondary (judgement) | confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| A | bipolar | 0.0919 | 0.0804 | -1.2344 | undetermined | -0.6172 | undetermined | ok | degenerate | high |
| A | depression | 0.0084 | 0.0022 | -0.0257 | certain-ok | -0.0129 | certain-ok | ok | admissible | certain |
| A | eating disorder | 0.0244 | 0.0046 | -0.2014 | undetermined | -0.1007 | undetermined | ok | degenerate | low |
| A | schizophrenia | 0.0551 | 0.0262 | -0.6974 | undetermined | -0.3487 | undetermined | ok | degenerate | high |
| B | bipolar | 0.2848 | 0.0804 | -3.9000 | certain-degenerate | -1.9500 | certain-degenerate | degenerate, robust | degenerate | certain |
| B | depression | 0.0152 | 0.0022 | -0.1075 | undetermined | -0.0538 | certain-ok | ok | admissible | low |
| B | eating disorder | 0.0506 | 0.0046 | -0.4863 | undetermined | -0.2432 | undetermined | ok | degenerate | high |
| B | schizophrenia | 0.1524 | 0.0262 | -1.8403 | certain-degenerate | -0.9202 | certain-degenerate | degenerate, robust | degenerate | certain |
| F1 | bipolar | 0.0647 | 0.0172 | -0.7667 | undetermined | -0.3919 | undetermined | ok | degenerate | high |
| F1 | depression | 0.0411 | 0.0017 | -0.5662 | undetermined | -0.2840 | undetermined | ok | degenerate | high |
| F1 | eating disorder | 0.0282 | 0.0042 | -0.2787 | undetermined | -0.1415 | undetermined | ok | degenerate | high |
| F1 | schizophrenia | 0.0460 | 0.0106 | -0.5170 | undetermined | -0.2638 | undetermined | ok | degenerate | high |
| F2 | bipolar | 0.0615 | 0.0115 | -0.7593 | undetermined | -0.3854 | undetermined | ok | degenerate | high |
| F2 | depression | 0.0403 | 0.0023 | -0.5086 | undetermined | -0.2554 | undetermined | ok | degenerate | high |
| F2 | eating disorder | 0.0289 | 0.0054 | -0.2814 | undetermined | -0.1434 | undetermined | ok | degenerate | high |
| F2 | schizophrenia | 0.0467 | 0.0118 | -0.5204 | undetermined | -0.2661 | undetermined | ok | degenerate | high |
| F3 | bipolar | 0.1026 | 0.0365 | -1.7534 | undetermined | -0.8950 | undetermined | degenerate, borderline | degenerate | high |
| F3 | depression | 0.0411 | 0.0017 | -0.5662 | undetermined | -0.2840 | undetermined | ok | degenerate | high |
| F3 | eating disorder | 0.0284 | 0.0046 | -0.2838 | undetermined | -0.1442 | undetermined | ok | degenerate | high |
| F3 | schizophrenia | 0.0464 | 0.0113 | -0.5263 | undetermined | -0.2688 | undetermined | ok | degenerate | high |
| L1 | bipolar | 0.3110 | 0.1360 | -7.2051 | certain-degenerate | -3.6706 | certain-degenerate | degenerate, robust | degenerate | certain |
| L1 | depression | 0.0411 | 0.0017 | -0.5662 | undetermined | -0.2840 | undetermined | ok | degenerate | high |
| L1 | eating disorder | 0.0291 | 0.0057 | -0.2984 | undetermined | -0.1521 | undetermined | ok | degenerate | high |
| L1 | schizophrenia | 0.0477 | 0.0134 | -0.5536 | undetermined | -0.2835 | undetermined | ok | degenerate | high |
| R0 | bipolar | 0.2457 | 0.0312 | -3.7034 | certain-degenerate | -1.8673 | certain-degenerate | degenerate, robust | degenerate | certain |
| R0 | depression | 0.0781 | 0.0415 | -0.8758 | undetermined | -0.4586 | undetermined | ok | degenerate | high |
| R0 | eating disorder | 0.0509 | 0.0321 | -0.6184 | undetermined | -0.3253 | undetermined | ok | degenerate | high |
| R0 | schizophrenia | 0.1525 | 0.0886 | -2.6202 | undetermined | -1.3544 | undetermined | degenerate, borderline | degenerate | high |

**Scoring.**

- **Primary:** certain cells, as above.
- **Secondary:** each undetermined cell is scored **held**, **falsified** (the opposite outcome) or **split**. High-confidence and low-confidence cells are reported separately.
- **D-042 / D-047:** each of their "ok" cells is a direct test. It is **confirmed** if admissible and **overturned** if degenerate. This settles what the amendments left open.

### 3. Prediction 2: the α at which each condition flips

δ_ci does not depend on α (equation (21)).

- **Primary (closed form).** The flip must lie in `[δ_ci − D̂_max, δ_ci − Δ̂_min]`. A measured flip outside that bracket is an implementation error.
- **D-042's implied flip** is at or below δ_ci.
- **Secondary (judgement).** Central `δ_ci − 0.5 D̂_max + P_k + 0.7 (K − 1)(δ̂* + ζ̂^upp)`. Low end: the same with 1.0 and 0.4 in place of 0.5 and 0.7.

Values of 1 or more mean no α < 1 rescues the condition.

| region | condition | guaranteed bracket, ζ̂ = δ̂* | D-042 implies | secondary central (low end), ζ̂ = δ̂* | secondary central, ζ̂ = 0 |
|---|---|---|---|---|---|
| A | bipolar | [0.011, 1.326] | ≤ 0.092 | 0.92 (≥ 0.51) | 0.48 |
| A | depression | [0.006, 0.034] | ≤ 0.008 | 0.03 (≥ 0.02) | 0.02 |
| A | eating disorder | [0.020, 0.226] | ≤ 0.024 | 0.16 (≥ 0.10) | 0.09 |
| A | schizophrenia | [0.029, 0.752] | ≤ 0.055 | 0.53 (≥ 0.31) | 0.29 |
| B | bipolar | [0.204, 4.185] | ≤ 0.285 | 2.97 (≥ 1.76) | 1.61 |
| B | depression | [0.013, 0.123] | ≤ 0.015 | 0.09 (≥ 0.06) | 0.05 |
| B | eating disorder | [0.046, 0.537] | ≤ 0.051 | 0.39 (≥ 0.24) | 0.22 |
| B | schizophrenia | [0.126, 1.993] | ≤ 0.152 | 1.43 (≥ 0.86) | 0.78 |
| F1 | bipolar | [0.047, 0.831] | ≤ 0.065 | 0.60 (≥ 0.36) | 0.34 |
| F1 | depression | [0.039, 0.607] | ≤ 0.041 | 0.44 (≥ 0.27) | 0.24 |
| F1 | eating disorder | [0.024, 0.307] | ≤ 0.028 | 0.22 (≥ 0.14) | 0.13 |
| F1 | schizophrenia | [0.035, 0.563] | ≤ 0.046 | 0.41 (≥ 0.25) | 0.23 |
| F2 | bipolar | [0.050, 0.821] | ≤ 0.062 | 0.59 (≥ 0.36) | 0.33 |
| F2 | depression | [0.038, 0.549] | ≤ 0.040 | 0.40 (≥ 0.24) | 0.22 |
| F2 | eating disorder | [0.023, 0.310] | ≤ 0.029 | 0.22 (≥ 0.14) | 0.13 |
| F2 | schizophrenia | [0.035, 0.567] | ≤ 0.047 | 0.41 (≥ 0.25) | 0.23 |
| F3 | bipolar | [0.066, 1.856] | ≤ 0.103 | 1.32 (≥ 0.79) | 0.72 |
| F3 | depression | [0.039, 0.607] | ≤ 0.041 | 0.44 (≥ 0.27) | 0.24 |
| F3 | eating disorder | [0.024, 0.312] | ≤ 0.028 | 0.23 (≥ 0.14) | 0.13 |
| F3 | schizophrenia | [0.035, 0.573] | ≤ 0.046 | 0.41 (≥ 0.25) | 0.23 |
| L1 | bipolar | [0.175, 7.516] | ≤ 0.311 | 5.33 (≥ 3.14) | 2.85 |
| L1 | depression | [0.039, 0.607] | ≤ 0.041 | 0.44 (≥ 0.27) | 0.24 |
| L1 | eating disorder | [0.023, 0.327] | ≤ 0.029 | 0.24 (≥ 0.15) | 0.13 |
| L1 | schizophrenia | [0.034, 0.601] | ≤ 0.048 | 0.43 (≥ 0.26) | 0.24 |
| R0 | bipolar | [0.214, 3.949] | ≤ 0.246 | 2.83 (≥ 1.71) | 1.55 |
| R0 | depression | [0.037, 0.954] | ≤ 0.078 | 0.68 (≥ 0.41) | 0.39 |
| R0 | eating disorder | [0.019, 0.669] | ≤ 0.051 | 0.48 (≥ 0.29) | 0.27 |
| R0 | schizophrenia | [0.064, 2.773] | ≤ 0.152 | 1.97 (≥ 1.17) | 1.08 |

**Measurement (step 9's sweep).** α ∈ {0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 0.95}, 100 runs per α, primary ζ̂, conservative variant. The measured flip α̂_k is the smallest grid α at which τ̂_k < 1 in at least 50 runs, or "none".

**Scoring, per cell.**

- **Primary:** α̂_k is inside the guaranteed bracket, to the grid's resolution. Otherwise it is a bug.
- **D-042 held:** α̂_k is at or below the first grid point ≥ δ_ci.
- **Secondary held:** α̂_k is at or above the last grid point ≤ the low end.

None of these can be rescued afterwards by moving the grid.

### 4. Prediction 3: what the closed form cannot give

**(a) Mean set size for truly bipolar test points, region A, conservative variant (secondary, a judgement: the closed form leaves bipolar, eating disorder and schizophrenia undetermined in region A, so it gives no size).** Predicted **3.1**; held if in [2.9, 3.3]. Reasoning: bipolar, eating disorder and schizophrenia are each predicted in every set under the secondary calls, and depression is admitted for a minority of bipolar posts. D-042's reading ("non-degenerate region A") implies at most about 1.6; a result at or below 1.6 confirms D-042 and falsifies this prediction. A result between 1.6 and 2.9 is "partly missed" and is reported per condition. Optimistic variant: predicted identical to arm (a), with thresholds equal in at least 95 of 100 runs for bipolar, eating disorder and schizophrenia.

**(b) Sesia et al.'s Assumptions 2, 3 and 5**, measured on each region's reference sample, per condition (28 cells).

- **Assumption 2** (densities of F̃^k_l bounded, and `f_min = min_k inf_{t∈(0,1)} f̃^k_k(t) > 0`). *Predicted:* the continuity and boundedness part holds. The `f_min > 0` part **fails** in every cell, because softmax scores have vanishing density as ŝ → 0 and ŝ → 1. *Operational test:* a 200-bin histogram of `ŝ(X, k)` given Ỹ = k on (0, 1); the part fails if the emptiest bin has density below 1% of the median bin.
- **Assumption 3** (`max_{l≠k} F̃^k_l(t) ≤ F̃^k_k(t)` for all t). *Predicted:* **holds** in all 28 cells. *Test:* `sup_t [max_{l≠k} F̃^k_l(t) − F̃^k_k(t)] ≤ 0.005` over the reference sample's scores.
- **Assumption 5** (equation (23)). As t → 0⁺, both CDFs in Δ_k vanish, so `inf_{t∈(0,1)} Δ_k(t) ≤ 0`, and Assumption 5 can hold only where the right-hand side of (23) is ≤ 0. Computed now, with the multiplier `1/√n* + √((log 2K + log n*)/2)` and the width term `(K − 1)(δ̂*_k + |ζ̂^upp_k|)` as printed, the right-hand side is positive in **27 of 28 cells**. The exception is region A depression, at −0.066; the next closest is region B depression at +0.023. Assumption 5 therefore fails in 27 cells by arithmetic. Under the ζ̂ = 0 sensitivity it fails in 26: region B depression drops to −0.031. *Predicted:* it holds in region A depression, where the test is `min_t Δ_dep(t) ≥ −0.066`. Consequence: Theorem 5's coverage upper bound, the guarantee that Algorithm 2 is not over-conservative, is unavailable everywhere except region A depression.

**(c) The D-032 head-to-head at region A width**, run in two orientations. **O1:** truth `T_HOC_FINE_TUNED`, arm (b) given the cleanlab point. **O2:** truth `T_CLEANLAB`, arm (b) given the HOC point. In both, arm (c) is given the region A box. Scored on bipolar and schizophrenia (D-032), conservative variant of record.

- **(c) WIN** for a condition: (c) is valid and not degenerate, and either (b) is invalid or (c)'s mean set size (over truly-k test points) is at least 0.05 below (b)'s.
- **WIN BY ABSTENTION:** (c) is valid only because it is degenerate, and (b) is invalid. Recorded, **not** counted as a win.
- **LOSS:** (b) is valid, and (c) is either invalid or at least 0.05 larger.
- **TIE:** otherwise.
- **The head-to-head is WON** only if (c) wins at least one of {bipolar, schizophrenia} in both orientations and loses neither.

*Predicted:*

- **O1:** (b) under-covers bipolar, because the cleanlab point overstates bipolar's inflow and over-corrects (low confidence), while (c) is degenerate. Result: win by abstention on bipolar and probably on schizophrenia.
- **O2:** (b) is valid, because a point that understates the noise is conservative, as in step 6, while (c) is degenerate. Result: **loss** on both conditions.
- **Overall:** the head-to-head is **not won**.

### 5. Prediction 4: Proposition 3 and the optimistic variant (required by D-049)

- **As stated:** fails in 100 of 100 runs in every cell. This is deterministic: the bound `δ_ci − (1 − α)/n_k` is positive in every cell, and `inf_t Δ_k ≤ 0`.
- **At the realised threshold:** fails in at least 90 of 100 runs in every cell.
- **Optimistic coverage:** nevertheless **valid** in every cell. Inflow into a label makes standard conformal conservative (Corollary 1), and the optimistic variant is predicted to reduce to standard conformal wherever the conservative one degenerates.
- **Scoring:** a cell where the optimistic variant is invalid falsifies this prediction, and it also shows the unverifiable condition costing coverage, which D-049 asked to measure.

### 6. What each outcome would mean for C2's scope

- **F1 and F2 come out degenerate (secondary calls held): width, not mass, is binding.** Under the conservative Algorithm 2 at the C0 counts, any V-interval wider than roughly `(α − δ_ci)/(0.7 · 6) ≈ 0.01 to 0.02` in a rare condition's row degenerates that condition. An elicited region would have to be almost a point. That contradicts C2's premise of a region rather than a point, for the arm of record. C2 then reports D-032's characterisation branch with width as the binding axis. The ζ̂ = 0 sensitivity shows how much a structured, uniform region would recover. Any route forward (such a region, larger calibration sets, or the optimistic variant with its unverifiable condition) needs its own decision entry.
- **F1 and F2 come out admissible (secondary calls falsified).** The width terms were small at the thresholds that matter. D-047's "ok" verdicts are confirmed in practice, though still not guaranteed by the closed form, and C2 proceeds as D-047 scoped it. Report the miss on the scoreboard.
- **Split results.** Report per condition. C2's claim is restricted to the conditions that are admissible.
- **Prediction 2.** Where α̂_k exceeds δ_ci, D-042's flip formula fails as a predictor of admissibility, though it still holds as a sufficient test for degeneracy. Where α̂_k is at or below δ_ci, the width terms were negligible there.
- **Prediction 3(c), head-to-head not won.** As D-032 anticipated, the headline becomes the characterisation: where the coupling stops working, now with region width as a named cause. Won: C2 is a method at region A width, the stronger claim.
- **Prediction 3(b).** Assumption 5 failing broadly is itself reportable: Algorithm 2 retains Theorem 4's validity but loses Theorem 5's tightness on this problem.

### Evidence

- Arithmetic from the functions named under Decision, run 2026-10-26 on `NOISY_PRIOR` and `CALIBRATION_COUNTS`, with `c(n_k)` from `expected_sup_deviation`.
- Equations (20) to (25) and Assumptions 2 to 5 from Sesia et al. (2025), read in the local copy. Assumption 5 was cross-read in arXiv:2309.05092 v2, where it is numbered (24) and its asymptotic form (25).
- The closed-form bracket follows from equation (20) with the CDFs in [0, 1]; the derivation is in Context.
- The heuristic factors (0.7 on the width terms, 0.5 on term one) are a judgement from the synthetic scorer's construction, not a measurement. They drive only the secondary tier.
- Independent check of equation (20): the subagent report summarised in Context, from the arXiv LaTeX source (labels `eq:delta-hat-k-epsilon-bound`, `eq:def-zeta-k`).

### Consequences

- Arm (c) runs only after this entry is committed. The tables above are the record the results are scored against.
- D-042, D-047 and `c2-elicitation-scoping.md` §0 and §3 carry amendments of this date recording the derivation error. The run settles which of their "ok" verdicts hold in practice.
- The reversal scoreboard gains one pre-registered prediction set, scored per section.

### Links

Tests D-042 and D-047. Implements roadmap step 7. Depends on D-028, D-029, D-031, D-032, D-048 and D-049. Results to D-046.

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
| 3 | Verify the `ρ = (Pᵀ)⁻¹ρ̃` derivation. **Numerical check passed 2026-10-26** (D-028 amendment: unbiased recovery with calibrated error on synthetic data with known T); the algebra and the novelty question still need the supervisor | D-028, D-031 | Soshan + supervisor |
| 4 | Verify the prevalence arithmetic in D-018 against Zhu et al.'s Theorem 2. **Counts half CLOSED 2026-08-14** (D-018 amendment: 146,400 raw vs 139,898 deduplicated, both correct, provenance recorded); the Theorem 2 formula check itself is still open | The strongest result in the project | Soshan |
| 5 | Decide prune vs no-prune | D-031 | Soshan + supervisor |
| 6 | Source every elicited range to a specific clinical study. **Re-scoped 2026-10-08 (D-047):** the binding quantity is the worst-case PPV of each rare label (about 0.78 for bipolar at α = 0.10), not the misdiagnosis directions. Non-binding entries are partly sourced; the binding ones have no subreddit-level study | The main attack on C2 | Soshan |
| 7 | Read Bortolotti et al. (arXiv:2501.18060) in full | D-029 | Soshan |
| 8 | Read Ding et al. (arXiv:2306.09335) in full | One leg of C3's convergence table is from a snippet | Soshan |
| 9 | Aich et al. access outcome | D-009 | External |
| 10 | Chapter 2 (Literature Review) draft | Next chapter | Soshan |
| 11 | Re-verify novelty before camera-ready | D-017; the noisy-conformal area published 4+ items in 2024–25 | Soshan |
| 12 | ~~One MentalRoBERTa fine-tune, identical protocol, splits and seed, reported as a Milestone 0 baseline replication (not a C1 arm)~~ **CLOSED 2026-09-27** (D-045): 0.8841 vs 0.8782 macro-F1, splits fingerprint-matched, MentalBERT retained | was D-011 | done |
| 13 | Read the PDFs behind D-047. **Fetched sources checked 2026-10-26:** Zimmerman et al. (2010), Ernala et al. (2019) and Dagani et al. (2017), with Birnbaum et al. (2017), checked against their full texts (`Docs/clinical-elicitation-matrix.md`); every load-bearing claim supported. **Still unread:** Judd et al. (2002, 2003), Etchecopar-Etchart et al. (2021) and Hudson et al. (2007), not yet accessed | D-033 | Soshan |
| 14 | Decide whether to elicit label PPVs directly (Sesia's direction) rather than T, and with what protocol | D-047, D-028 | Soshan + supervisor |

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

**Amendment (2026-10-08, D-047): these three claims do not bind.** The budget sees posts *entering* a label, not posts leaving it. Bipolar → depression (claim 1) changes only depression's row, and is affordable up to about 60%. What exhausts bipolar's budget is depression → bipolar, near 1% of depressed authors' posts, equivalently a worst-case PPV for r/bipolarreddit below about 0.78. The sources needed are therefore on the validity of the rare labels, and the closest located (Zimmerman et al., 2010; Birnbaum et al., 2017; Ernala et al., 2019) sit at 0.22 to 0.43 in other populations. See `Docs/c2-elicitation-scoping.md` §4.

---

## Key references, verified by reading the full paper

| Reference | Role |
|---|---|
| Sesia, M., Wang, Y.X.R. and Tong, X. (2025) 'Adaptive conformal classification with noisy labels', *JRSSB* 87(3), 796–815, doi:10.1093/jrsssb/qkae114 | C2's foundation. Theorem 4 is the slot being filled. |
| Penso, C., Goldberger, J. and Fetaya, E. (2025) 'Conformal prediction of classifiers with many classes based on noisy labels', COPA, PMLR 266, 1–14 | The elicitable parameterisation (D-028). States that for a general noise matrix "all finite sample correction terms are not effective." |
| Zhu, Z., Song, Y. and Liu, Y. (2021) 'Clusterability as an alternative to anchor points when learning with noisy labels', ICML, PMLR 139, 12912–12923 | Killed D-024. Theorem 2 gives D-018's arithmetic. |
| Zhu, Z., Wang, J. and Liu, Y. (2022) 'Beyond images: label noise transition matrix estimation for tasks with lower-quality features', ICML, PMLR 162 | Rescued the project. Evidence for D-030. |
| Liu, Y., Cheng, H. and Zhang, K. (2023) 'Identifiability of label noise transition matrix', ICML, PMLR 202, 21475–21496 | Underpins D-018. Read in full 2026-08-09: it documents **four** identifiability routes, not three; the fourth (disentangled informative features, Theorem 5.5) prompted D-040. |
| Ji, S., Zhang, T., Ansari, L., Fu, J., Tiwari, P. and Cambria, E. (2022) 'MentalBERT: publicly available pretrained language models for mental healthcare', LREC 2022, 7184–7190 | The encoder's provenance. Section 2.2 supplies the pretraining corpus for D-044; section 3.1 records that SWMH uses weak labels; section 6 states that the model predictions are not psychiatric diagnoses. Read in full 2026-08-18, correcting two prior claims (see the scoreboard). |
| Zimmerman, M., Ruggero, C.J., Chelminski, I. and Young, D. (2010) 'Psychiatric diagnoses in patients previously overdiagnosed with bipolar disorder', *J Clin Psychiatry* 71(1), 26–31 | Load-bearing for D-047: the label PPV of 0.43 is derived from it. Claims checked against the full text in NotebookLM, 2026-10-26; all supported (`Docs/clinical-elicitation-matrix.md`, claims 1 to 8). |
| Ernala, S.K. et al. (2019) 'Methodological gaps in predicting mental health states from social media: triangulating diagnostic signals', CHI 2019, 1–16, doi:10.1145/3290605.3300364 | D-047: 153 of 635 appraised genuine. Checked against the PDF in NotebookLM, 2026-10-26 (claims 14 to 16). Its "N = 146" is Birnbaum et al.'s set, not a second count. |
| Birnbaum, M.L., Ernala, S.K., Rizvi, A.F., De Choudhury, M. and Kane, J.M. (2017) 'A collaborative approach to identifying social media markers of schizophrenia by employing machine learning and clinical appraisals', *JMIR* 19(8), e289 | D-047: 146 of 671 appraised genuine. Claim 10 checked in NotebookLM; claims 11 to 13 confirmed in the publisher's JATS XML (Methods, Clinician Appraisal), 2026-10-26. |
| Dagani, J. et al. (2017) 'Meta-analysis of the interval between the onset and management of bipolar disorder', *Can J Psychiatry* 62(4), 247–258, doi:10.1177/0706743716656607 | Non-binding entry only (D-047). Checked against the PDF in NotebookLM, 2026-10-26 (claims 20 and 21). |

## Key references, cited but NOT yet read in full

| Reference | Risk |
|---|---|
| Ding et al. (2023, NeurIPS), 'Class-conditional conformal prediction with many classes', arXiv:2306.09335 | One leg of C3's convergence table. |
| Bortolotti et al. (2025), arXiv:2501.18060 | Bears on D-029. |
| Einbinder et al. (2024), *JMLR* 25(328), 1–66 | Context-setting only; lower risk. |
| Zimmerman et al. (2008); Hirschfeld et al. (2003); Jagfeld et al. (2021) | Abstract only. Do not characterise beyond the abstract. |

---

## Reversal scoreboard

| Claims made from abstracts or snippets | 6 |
| Of those, materially wrong | **6** |
| Claims made after reading the full paper | 8 |
| Of those, materially wrong | **1** (D-042's `0 ≤ Δ̂_ci`, inherited by D-047) |
| Of those, not yet tested against a run | 1 (D-042's interface and sufficient degeneracy test) |
| Claims made from full text read through a fetch tool (excerpts, not the PDF) | 18 |
| Of those, materially wrong | **1** (the Ernala et al. N = 146 note) |
| Pre-registered predictions recorded before a run | 4 |
| Of those, held in full | 2 (D-035's rule, D-036/D-037) |
| Of those, falsified or partly missed | 2 (D-034 falsified, D-040 partly missed) |

The fetch-tool row was added on 2026-10-26 from the audit recorded in
`Docs/clinical-elicitation-matrix.md`. It counts the 17 audited claims from the four
fetched sources behind D-047 (claims 1 to 8, Zimmerman et al. 2010; 10 to 13, Birnbaum
et al. 2017; 14 to 16, Ernala et al. 2019; 20 and 21, Dagani et al. 2017), plus the
interpretation that Ernala et al. give both 153 and 146 genuine self-reporters. Claims
9, 19 and 22 state only what an abstract says and are excluded. The one wrong claim is
that interpretation: the 146 is Birnbaum et al.'s count. It was not load-bearing (the
PPV used 153) and the audit caught it before it reached the thesis.

NotebookLM returned PARTIAL or UNSUPPORTED for Birnbaum claims 11 to 13, all three of
which are verbatim in the publisher's JATS XML under *Methods, Clinician Appraisal*.
Retrieval misses happen in both directions, so a negative NotebookLM verdict is checked
against the source before it changes anything.

The D-045 MentalRoBERTa replication is deliberately absent from the prediction block. Its
only rule was qualitative ("small gap" versus "large gap") with no threshold fixed before the
run, so it does not qualify as a pre-registered prediction and is not counted as held.

**This table is the argument for D-033.** Keep it updated. If it ever shows a wrong claim made after reading a full paper, that is worth knowing too.

The fifth and sixth abstract-derived claims were added on 2026-08-18, both about
MentalBERT and both taken from secondary summaries rather than from Ji et al. (2022):
(i) that its domain-adaptive pretraining used next sentence prediction alongside
masked language modelling, and (ii) that its crawl window is documented and overlaps
Low et al.'s collection windows. Both fail against the paper, which uses MLM only and
reports no collection dates at all. Neither claim had become load-bearing before it
was checked: (i) never entered an entry, and (ii) would have made D-044 assert
contamination the primary source cannot support.

The sixth full-paper claim is D-040's, from Liu, Cheng and Zhang (2023): that there is
a fourth identifiability route and that HOC's accuracy depends materially on the
feature extractor. D-041 bore it out, and more strongly than expected: the bipolar
diagonal moves from 0.5409 to 0.9534 across three encoders on identical data.

The seventh is D-042's, from Sesia, Wang and Tong (2025): that Algorithm 2 consumes an
interval box on the off-diagonals of `V = M⁻¹`, and that its degenerate case is a
closed-form inequality. It is counted separately as untested because no run has checked
it. D-043 should record the α at which each condition is predicted to flip, so that the
simulation can either confirm it or move it into the wrong column.

**The eighth, and the first full-paper claim found materially wrong (2026-10-26).**
D-042 §3 stated `0 ≤ Δ̂_ci_k(S_(n_k)) ≤ D̂_max`, and its "survives" verdicts rested on
the lower bound. Equation (20), in the same paper read in full, subtracts two width terms
that make Δ̂_ci negative once the box has width (D-042 amendment of this date, with a
worked instance). It is split out from the seventh claim because the two parts have
different status. The interface, and the upper bound behind the robust degenerate
verdicts, still stand untested. The lower bound is wrong, and that is settled by the
equation, not by a run. D-047 inherited the error: its frontier, its PPV floor of about
0.78, and every "ok" verdict became necessary conditions only. The source was
read in full, and the bound was derived from equation (20)'s first term alone; the
record does not show whether the other two terms were overlooked or judged negligible. It was caught while writing D-043's pre-registration, before any
arm (c) run, and confirmed independently from the arXiv source.

**The second block is the more useful one, and it is deliberately separate.** A project
that only recorded predictions it got right would be recording nothing. Two of four
pre-registered predictions did not hold as stated, both are written up in full
(D-034, D-041), and in both cases the pre-registration is what stopped the result being
reinterpreted after the fact.
