# C2 elicitation scoping: what the budget asks of the clinical literature

_Date: 2026-10-08. Answers step 3b of `progress-and-roadmap.md` §5 and reframes open item 6.
Code: `src/conformal/elicited_region.py`, reproduced by
`python -m src.conformal.elicited_region`; `tests/test_conformal_elicited_region.py` pins
every number below. Decision entry: D-047._

_This is a derivation, not a run. Like D-042 it has not been checked against a
simulation, and it rests on D-028's conversion `ρ = (Tᵀ)⁻¹ρ̃`, which open item 3 still
lists as unverified._

---

## 0. The answer

1. **A region can fit all four budgets.** At α = 0.10 and n\* = 481 it needs, at its
   pessimistic end, roughly: depression into each rare label at most 0.5% (the
   boundary sits near 1%), rare into rare at most about 0.5% to 2%, and leakage into
   r/depression at most 40% (60% for bipolar on its own).
2. **In clinical terms, the bipolar condition is a floor on the bipolar label's
   positive predictive value (PPV):** the worst-case share of r/bipolarreddit posts
   written by someone who truly has bipolar disorder must be about three-quarters or
   more. A region whose worst case is 0.763 degenerates (borderline); one at 0.873
   passes.
3. **The three directional claims in open item 6 constrain entries the budget barely
   sees.** Bipolar's budget is untouched by bipolar posts landing in r/depression, the
   direction diagnostic delay predicts. What it sees is posts *entering* the rare
   labels from other conditions, above all from depression.
4. **The only measured figures for comparable labels sit well below the floor:** 0.43
   for a self-reported prior bipolar diagnosis in an outpatient clinic (Zimmerman et
   al., 2010), and 0.22 to 0.24 for schizophrenia self-disclosure on Twitter (Birnbaum
   et al., 2017; Ernala et al., 2019). No study found measures it for subreddit
   membership. A region reflecting a PPV of 0.45 (L1 below) reaches a bipolar V mass of
   1.49, the same scale as C1's region B (1.354).

**Projected outcome at the C0 operating point.** Depression and eating disorder are
admissible across every region tested. Bipolar is inadmissible under any region the
literature located here supports. Schizophrenia depends on an elicited PPV for
r/schizophrenia that the literature does not pin down. This is D-032's characterisation
branch in its per-condition form, unless a structured elicitation (§8) can defend a
bipolar PPV floor near 0.8.

---

## 1. Which entries of T the budget sees

By Bayes, `M_kl = T_lk · ρ_l / ρ̃_k`. Row k of M, and therefore row k of V = M⁻¹, is
built from the posts that **carry label k**. Three consequences follow, and the first
is the one open item 6 was missing.

- **Posts leaving condition k for another label do not enter row k at all.** A truly
  bipolar author posting in r/depression changes which label the post carries, not who
  wrote the posts labelled bipolar. With bipolar to depression at 0.6 and nothing else,
  M's bipolar row stays exactly (1, 0, 0, 0)
  (`test_bipolar_row_of_v_ignores_leakage_out_of_bipolar`).
- **Posts entering label k from condition l are amplified by `ρ_l / ρ̃_k`.** For
  depression into bipolar that ratio is about 0.81 / 0.038 ≈ 21. One per cent of
  depressed authors' posts in r/bipolarreddit is about a fifth of r/bipolarreddit.
- **Leakage into r/depression is diluted by the same ratio run backwards**, so it
  costs depression's row little.

Each off-diagonal entry on its own, as a point matrix with no interval width:

| Entry (true → label) | Breaks a budget at | Budget broken |
|---|---|---|
| depression → bipolar | **0.0130** | bipolar |
| depression → schizophrenia | **0.0190** | schizophrenia |
| depression → eating disorder | **0.0350** | eating disorder |
| eating disorder → bipolar | 0.0980 | bipolar |
| eating disorder → schizophrenia | 0.1410 | schizophrenia |
| schizophrenia → bipolar | 0.1620 | bipolar |
| bipolar → schizophrenia | 0.2930 | schizophrenia |
| schizophrenia → eating disorder | 0.3525 | eating disorder |
| bipolar → eating disorder | 0.4360 | eating disorder |
| eating disorder → depression | 0.7375 | depression |
| schizophrenia → depression | 0.8335 | depression |
| bipolar → depression | 0.8765 | depression |

These are optimistic limits. A real region has many non-zero entries and width, and
breaches earlier (§2, §3).

---

## 2. Jointly, the cheap entries stop being cheap

Region **R0** takes open item 6's three directions wide (leakage into r/depression up to
60% for bipolar and 40% for the other two), allows rare-to-rare inflow up to 10%, and
sets depression-to-rare inflow to **zero**. Bipolar still degenerates robustly:

| R0 | V mass | budget | δ_ci | margin | verdict |
|---|---|---|---|---|---|
| bipolar | 1.147 | 0.378 | 0.2457 | +0.1457 | degenerate, robust |
| schizophrenia | 0.688 | 0.411 | 0.1525 | +0.0525 | degenerate, borderline |
| eating disorder | 0.182 | 0.441 | 0.0509 | −0.0491 | ok |
| depression | 0.381 | 0.497 | 0.0781 | −0.0219 | ok |

The mechanism: wide leakage into r/depression raises the recovered clean prevalence of
the rare conditions (40% leakage turns a 0.054 noisy schizophrenia share into a clean
share near 0.09), and that larger prevalence multiplies every rare-to-rare inflow into
the bipolar label. Clinically plausible leakage into depression is affordable only if
rare-to-rare inflow stays small.

---

## 3. The feasibility frontier

Five constructed regions. **None of them is an elicitation**; they map where the
boundary sits. All lower bounds are zero except in R0, and each box is propagated to V
by vertex enumeration with an interior check (no interior sample escaped any box).

| Region | Leakage into r/depression | Rare → rare | Depression → bipolar | Depression → scz, ED | Bipolar V mass | Bipolar δ_ci | Verdict | Worst-case bipolar PPV |
|---|---|---|---|---|---|---|---|---|
| F1 | ≤ 40% | ≤ 0.5% | ≤ 0.5% | ≤ 0.5% | 0.192 | 0.0647 | ok | 0.873 |
| F2 | bipolar ≤ 60%, others ≤ 30% | ≤ 0.5% | ≤ 0.5% | ≤ 0.5% | 0.175 | 0.0615 | ok | 0.873 |
| F3 | ≤ 40% | ≤ 0.5% | **≤ 1%** | ≤ 0.5% | 0.392 | 0.1026 | degenerate, borderline | 0.763 |
| L1 | ≤ 40% | ≤ 0.5% | **≤ 2.4%** | ≤ 0.5% | 1.492 | 0.3110 | degenerate, robust | **0.450** |
| R0 | §2 | ≤ 10% | 0 | 0 | 1.147 | 0.2457 | degenerate, robust | 0.480 |

All four conditions pass in F1 and F2. In F3 and L1 only bipolar fails; depression,
eating disorder and schizophrenia stay admissible with margins of at least 0.05.

**A grid over the three region parameters** (vertex enumeration, α = 0.10). Each cell is
the largest depression-to-rare bound at which all four conditions pass; a dash means
none does.

| Leakage into r/depression \ rare → rare | 0 | 0.5% | 1% | 2% | 5% |
|---|---|---|---|---|---|
| 5% | 1% | 1% | 1% | 0.5% | 0.1% |
| 10% | 1% | 1% | 1% | 0.5% | 0 |
| 20% | 1% | 1% | 0.5% | 0.5% | none |
| 30% | 1% | 1% | 0.5% | 0.5% | none |
| 40% | 1% | 0.5% | 0.5% | 0.5% | none |
| 60% (all three) | none | none | none | none | none |

The 60% row fails on **depression's** budget: leakage that large from all three rare
conditions together pushes depression's row over 0.497. Bipolar alone can leak 60%
(F2). In the 40% row, 1% passes only when rare-to-rare inflow is zero; with
rare-to-rare at 0.5% the limit falls to 0.5%, and 1% into bipolar under those settings
is region F3's borderline failure.

Across the grid, the lowest worst-case bipolar PPV that passed was 0.744 and the
highest that failed was 0.780. **A floor of about 0.78 is sufficient; a floor below
about 0.74 is not.**

**Sensitivity to the noisy prior.** Re-running with the deduplicated shares from the
D-018 amendment (bipolar 0.0371 rather than 0.038) moves nothing material: depression
to bipolar at 0.8% passes (bipolar mass 0.315, δ_ci 0.0880) and at 1% fails borderline
(0.405, 0.1050).

---

## 4. What the literature supplies for the binding quantity

The binding quantity is the PPV of each rare label, equivalently the inflow into it.
Sources located, what each measures, and how far it transfers:

| Source | What it measures | Figure | Read status |
|---|---|---|---|
| Zimmerman, Ruggero, Chelminski and Young (2010) | 700 psychiatric outpatients (Rhode Island MIDAS project) interviewed with the SCID; asked whether previously diagnosed with bipolar disorder | 145 (20.7%) reported a prior diagnosis; 82 of those (56.6%) not confirmed by the SCID; 68 of the 82 (82.9%) had lifetime major depressive disorder; 90 patients received a SCID bipolar diagnosis | Full text on the publisher page, read through a fetch tool; figures quoted from its excerpts |
| Zimmerman, Ruggero, Chelminski and Young (2008) | The same project's first report | "Fewer than half" of self-reported prior diagnoses confirmed | Abstract only |
| Birnbaum, Ernala, Rizvi, De Choudhury and Kane (2017) | 671 randomly sampled Twitter users matching schizophrenia or psychosis self-disclosure phrases, appraised by clinicians | 146 genuine, 101 maybe, 424 not genuine (jokes, quotes); κ = 0.81 for genuine against not | Full text (JMIR, open access) |
| Ernala et al. (2019) | Three proxy signals for schizophrenia on Twitter (affiliation with a support organisation's account, self-report phrases, clinician-appraised self-reports) against 88 clinically diagnosed patients | 153 of 635 self-reporters appraised genuine (the paper's N = 146 is Birnbaum et al.'s set); a model trained on the affiliation proxy scores AUC 0.95 internally and 0.20 on patients | Full text (authors' copy), read through a fetch tool |
| Sekulić, Gjurković and Šnajder (2018) | Reddit users stating "I am diagnosed with bipolar" across five bipolar subreddits | 250 users checked, no false positives *as statements*; no subreddit PPV derivable | Full text (ACL Anthology) |
| Jagfeld et al. (2021) | About 20,000 Reddit users self-reporting a bipolar diagnosis | "Often report additional mental health diagnoses" | Abstract only |

The verified record of these sources, with a verdict per claim from the 2026-10-26 audit,
is `Docs/clinical-elicitation-matrix.md`; the read status above is as of 2026-10-08.

Derived from Zimmerman et al. (2010), not stated in it: of the 145 self-reports, 63 were
confirmed (145 − 82), so the label's PPV in that clinic was **63 / 145 = 0.43**; and of
the 90 SCID-confirmed cases, 63 had reported a prior diagnosis (0.70). Of the 610
patients without a SCID bipolar diagnosis, 82 (13.4%) carried the label, and at least
68 / 610 = 11.1% carried it alongside lifetime major depression.

**How far these transfer.** Four limits, each stated before an examiner raises it.

- **Direction and population.** These are PPVs, Sesia's direction, measured in their own
  populations. D-028's point applies: a PPV depends on prevalence, so the numbers do not
  carry over to Reddit as numbers. What carries over is the existence and rough scale of
  the channel: a diagnostic label for bipolar disorder attaches to many people whose
  structured-interview diagnosis is depression.
- **The data refute the clinic rate as a Reddit rate.** Transporting the 11% figure as
  depression to bipolar in T recovers a negative clean bipolar prevalence: 11% of
  depressed authors' posts would exceed the entire observed bipolar share
  (`test_clinic_rate_is_inconsistent_with_the_observed_bipolar_share`). The observed
  noisy marginals cap that entry at roughly ρ̃_bipolar / ρ_depression ≈ 0.047.
- **Most documented contamination comes from outside the four classes.** Birnbaum's
  "not genuine" class is jokes and quotes. Zimmerman's overdiagnosed group carries
  significantly higher rates of borderline personality disorder (24.4% against 6.1%),
  PTSD and impulse control disorders. Partners, carers and people asking whether they
  have a condition post in these communities too. A 4×4 contamination model has no row
  for "none of the four", so this mass cannot be represented; it either violates the
  model or must be assigned to a class by assumption. This is a named limitation of
  C2 alongside Assumption 1.
- **Phrases are not subreddits.** Twitter phrase-matching captures jokes in a way that
  posting in a support community rarely does, so Birnbaum's and Ernala's figures are a
  pessimistic analogue for r/schizophrenia, not an estimate of it. Equally, posting in a
  community requires no diagnosis at all, which phrase-matching on "I am diagnosed"
  does.

**The gap.** No study located measures the clinical validity of subreddit membership
for any of the four conditions. The literature can show the binding channel exists in
clinical populations; it cannot certify a Reddit PPV floor of 0.78, and the measured
analogues lean against it.

---

## 5. What the literature supplies for the non-binding entries

These justify wide leakage into r/depression, which the frontier tolerates up to 40%
(60% for bipolar alone).

| Source | Figure | Read status |
|---|---|---|
| Dagani et al. (2017), meta-analysis | 27 studies, 51 samples, 9,415 patients; pooled interval between onset and management of bipolar disorder 5.8 years (95% CI 4.8 to 6.8) | Full text on the publisher page, read through a fetch tool |
| Hirschfeld, Lewis and Vornik (2003), NDMDA survey | First 600 of 4,192 distributed questionnaires; 69% misdiagnosed, most often as unipolar depression; over a third waited 10 years or more | Abstract only |
| Judd et al. (2002, 2003), phase predominance | Not accessed (publisher access check) | **To read.** Do not quote figures before reading |
| Etchecopar-Etchart et al. (2021), comorbid major depression in schizophrenia | Not accessed | **To read** |
| Hudson et al. (2007), eating disorders in the NCS-R | Not accessed | **To read** |

Because these entries are cheap, their ranges can be set wide without costing the
budget, which is the behaviour open item 6 anticipated ("where the literature is thin,
the range goes wide"). The constraint is joint (§2): wide leakage is affordable only
while rare-to-rare inflow stays near 0.5% to 2%.

---

## 6. The operating point

The budget moves with α and with the calibration share, both reportable design
variables under D-042. Bipolar's budget, and the approximate PPV floor it implies,
`1 / (1 + budget)`. The approximation is slightly optimistic: at α = 0.10 it gives 0.73
where the grid in §3 puts the boundary between 0.744 and 0.780.

| α | C0 test split (bipolar n = 481) | 20% share (1,039) | 30% share (1,558) | 50% share (2,597) |
|---|---|---|---|---|
| 0.05 | 0.115 (PPV > 0.90) | 0.229 (> 0.81) | 0.306 (> 0.77) | 0.427 (> 0.70) |
| 0.10 | **0.379 (> 0.73)** | 0.603 (> 0.62) | 0.755 (> 0.57) | 0.993 (> 0.50) |
| 0.15 | 0.643 (> 0.61) | 0.977 (> 0.51) | 1.205 (> 0.45) | 1.560 (> 0.39) |
| 0.20 | 0.906 (> 0.52) | 1.351 (> 0.43) | 1.654 (> 0.38) | 2.126 (> 0.32) |

The clinic PPV of 0.43 becomes admissible only near α = 0.20 with a calibration share of
20% or more. That is the configuration `c2-interface.md` §6 recorded as a measured point
on the frontier and declined to recommend: 80% label-conditional coverage for a
screening instrument, bought by moving a fifth of the corpus out of training.

---

## 7. D-031 does not decide the bipolar verdict

Pruning against the observed proportions (D-025 option (b)) removes the part of a region
whose recovered prior leaves the simplex. For depression into bipolar that part begins
near 4.7%, while the budget is exhausted near 1%. Between those two points V is finite
and over budget, and pruning removes nothing there. **No region whose depression-to-
bipolar bound lies above about 1% becomes admissible by pruning.** D-031 can be settled
on its other merits (deterministic containment, the α_V = 0 corollary), with this
recorded as one fact the supervisor discussion does not need to weigh.

---

## 8. What can still move the outcome

For the supervisor meeting, in the order they bear on C2's claim:

1. **A structured expert elicitation of label PPVs, not of misdiagnosis rates.** Ask
   clinicians directly for a plausible range for "of posts in r/bipolarreddit, the share
   written by someone who meets criteria for bipolar disorder", and the same for
   r/schizophrenia and r/EDAnonymous, using an established protocol (for example the
   Sheffield Elicitation Framework, SHELF). The success criterion is a defensible
   pessimistic end of about 0.78 or higher for bipolar at α = 0.10. Any such elicitation
   has to engage Zimmerman et al. (2010) and Birnbaum et al. (2017), since an examiner
   will. Note that D-028 chose Penso's direction because PPVs depend on prevalence; a
   PPV elicited for these specific communities is anchored to their prevalence by
   construction, so that objection weakens here, but the change of direction should be
   recorded as a decision in its own right rather than slipped in.
2. **A descriptive audit of the corpus**, which needs no clean labels: the share of
   r/bipolarreddit and r/schizophrenia posts written by partners and carers or asking
   whether the author has the condition (lexical patterns), and author-level
   cross-posting among the four subreddits (author IDs are in the deduplication key).
   This bounds the closed-world violation in §4 and informs the elicitation. It does not
   estimate T, and should not be presented as doing so.
3. **The operating point** (§6): α and calibration share move the budget, at the costs
   already recorded.
4. **Report per-condition admissibility as the C2 result**, D-032's characterisation
   branch: C2 functions as a method for depression and eating disorder, for
   schizophrenia if a PPV floor near 0.76 is defensible, and the bipolar frontier is the
   finding. The contrast with C1 carries over intact: a literature-leaning elicited
   region (L1, 1.49) and C1's data-driven spread (region B, 1.354) fail bipolar's budget
   by similar factors, for different reasons.

---

## 9. Limitations of this scoping

- **A derivation, not a run.** D-043's pre-registration should carry the per-condition
  predictions here (which regions pass, and the α at which each condition flips) so the
  simulation can confirm or move them.
- **Rests on D-028's conversion**, still marked unverified (open item 3).
- **Region shape.** Independent intervals on the entries of T, then a box on V. Joint
  constraints a clinician might state (for example, a cap on total inflow into a label)
  would give a smaller set, but Algorithm 2 still takes a box on V, so the second
  boxing step remains.
- **The closed world and Assumption 1** (§4; `c2-interface.md` §8).
- **Source reading.** Zimmerman et al. (2010), Ernala et al. (2019) and Dagani et al.
  (2017) were first read as full-text web pages through a fetch tool that returns
  excerpts. These, with Birnbaum et al. (2017), were checked against the full texts in
  NotebookLM on 2026-10-26, and the volume and page numbers confirmed; every
  load-bearing claim was supported (verdicts in `Docs/clinical-elicitation-matrix.md`).
  Hirschfeld et al. (2003) and Zimmerman et al. (2008) remain abstract-only.

---

## 10. Reproducing

```bash
python -m src.conformal.elicited_region             # §1 and §3 tables
python -m pytest tests/test_conformal_elicited_region.py
```

The §3 grid and the §6 table were produced by the same functions with the parameters
shown; they are not pinned by tests.

---

## References

Birnbaum, M.L., Ernala, S.K., Rizvi, A.F., De Choudhury, M. and Kane, J.M. (2017) 'A
collaborative approach to identifying social media markers of schizophrenia by
employing machine learning and clinical appraisals', *Journal of Medical Internet
Research*, 19(8), e289.

Dagani, J., Signorini, G., Nielssen, O., Bani, M., Pastore, A., de Girolamo, G. and
Large, M. (2017) 'Meta-analysis of the interval between the onset and management of
bipolar disorder', *Canadian Journal of Psychiatry*, 62(4), pp. 247-258.
doi:10.1177/0706743716656607.

Ernala, S.K., Birnbaum, M.L., Candan, K.A., Rizvi, A.F., Sterling, W.A., Kane, J.M. and
De Choudhury, M. (2019) 'Methodological gaps in predicting mental health states from
social media: triangulating diagnostic signals', *Proceedings of the 2019 CHI
Conference on Human Factors in Computing Systems*, pp. 1-16.
doi:10.1145/3290605.3300364.

Hirschfeld, R.M.A., Lewis, L. and Vornik, L.A. (2003) 'Perceptions and impact of bipolar
disorder: how far have we really come? Results of the National Depressive and
Manic-Depressive Association 2000 survey of individuals with bipolar disorder',
*Journal of Clinical Psychiatry*. PMID 12633125. *Abstract only; volume and pages to
confirm.*

Jagfeld, G., Lobban, F., Rayson, P. and Jones, S.H. (2021) 'Understanding who uses
Reddit: profiling individuals with a self-reported bipolar disorder diagnosis',
*Proceedings of the Seventh Workshop on Computational Linguistics and Clinical
Psychology (CLPsych 2021)*. *Abstract only.*

Sekulić, I., Gjurković, M. and Šnajder, J. (2018) 'Not just depressed: bipolar disorder
prediction on Reddit', *Proceedings of the 9th Workshop on Computational Approaches to
Subjectivity, Sentiment and Social Media Analysis (WASSA)*. ACL Anthology W18-6211.

Zimmerman, M., Ruggero, C.J., Chelminski, I. and Young, D. (2008) 'Is bipolar disorder
overdiagnosed?', *Journal of Clinical Psychiatry*. PMID 18466044. *Abstract only.*

Zimmerman, M., Ruggero, C.J., Chelminski, I. and Young, D. (2010) 'Psychiatric
diagnoses in patients previously overdiagnosed with bipolar disorder', *Journal of
Clinical Psychiatry*, 71(1), pp. 26-31.
