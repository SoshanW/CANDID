# MentalRoBERTa replication: record of run

_Factual record of the D-011 / open item 12 replication: Milestone 0 rerun with
MentalRoBERTa in place of MentalBERT. For a plain-English walkthrough, see
[`roberta-replication-results-explained.md`](roberta-replication-results-explained.md).
Decision entry: D-045._

- **Artifacts analysed locally:** 2026-09-27 (`Models_roberta/`, copied from Drive; gitignored)
- **Environment:** Google Colab GPU, notebook section 9b; comparison recomputed locally on the CPU `.venv`
- **Status:** Robustness line under the C0 baseline table. Not a C1 arm.

---

## 1. Configuration

Identical to [`baseline-results.md`](baseline-results.md) §1 except for the encoder.

| Item | MentalBERT (C0 of record) | MentalRoBERTa (this run) |
|---|---|---|
| Base model | `mental/mental-bert-base-uncased` | `mental/mental-roberta-base` |
| Epochs / batch / LR / max length | 3 / 16 / 2e-5 / 256 | 3 / 16 / 2e-5 / 256 |
| Loss | unweighted cross-entropy | unweighted cross-entropy |
| fp16 / seed | on / 42 | on / 42 |
| Checkpoint | final step | final step |

**Seeds.** One training seed per encoder (42). No seed variance is available for either
encoder, so every uncertainty figure below is test-set resampling uncertainty only.

## 2. Split integrity (checked before reading any metric)

`train.py` rebuilds the split from `--seed` rather than reloading it, so identical splits
were an assumption until fingerprinted. SHA-256 of `pd.util.hash_pandas_object`, first 16 hex:

| Split | Rows | MentalBERT | MentalRoBERTa | |
|---|---|---|---|---|
| train | 111,892 | `90162cd1b3d6678b` | `90162cd1b3d6678b` | MATCH |
| val | 13,967 | `86ea56a8f2c55968` | `86ea56a8f2c55968` | MATCH |
| test | 14,039 | `256e8754aa0700a8` | `256e8754aa0700a8` | MATCH |

The two prediction files are also row-aligned (identical `text` and `true_condition` in
the same order), which is what licenses the paired tests in §5.

## 3. Headline metrics (held-out test set, 14,039 rows)

| Metric | MentalBERT | MentalRoBERTa | Δ (RoBERTa − BERT) |
|---|---|---|---|
| Accuracy | 0.9605 | 0.9634 | +0.0029 |
| **Macro-F1** | **0.8782** | **0.8841** | **+0.0060** |
| Macro precision | 0.9043 | 0.9068 | +0.0025 |
| Macro recall | 0.8552 | 0.8640 | +0.0088 |
| Weighted-F1 | 0.9596 | 0.9626 | +0.0030 |

## 4. Per-condition metrics (test set)

| Condition | Support | BERT P | BERT R | BERT F1 | RoBERTa P | RoBERTa R | RoBERTa F1 | ΔF1 | ΔF1 95% CI (row bootstrap) |
|---|---|---|---|---|---|---|---|---|---|
| depression | 11,351 | 0.9715 | 0.9860 | 0.9787 | 0.9748 | 0.9864 | 0.9806 | +0.0019 | [+0.0004, +0.0033] |
| eating_disorder | 1,421 | 0.9515 | 0.9388 | 0.9451 | 0.9557 | 0.9557 | 0.9557 | +0.0106 | [+0.0034, +0.0178] |
| schizophrenia | 786 | 0.8968 | 0.7850 | 0.8372 | 0.8862 | 0.7926 | 0.8368 | −0.0004 | [−0.0167, +0.0151] |
| bipolar | 481 | 0.7972 | 0.7110 | 0.7516 | 0.8107 | 0.7214 | 0.7635 | +0.0118 | [−0.0088, +0.0336] |

The condition ranking by F1 is identical under both encoders:
depression > eating_disorder > schizophrenia > bipolar.

## 5. Confusion matrices (rows = true, columns = predicted)

**MentalBERT**

| true ↓ / pred → | bipolar | depression | eating_disorder | schizophrenia |
|---|---|---|---|---|
| bipolar | 342 | 116 | 4 | 19 |
| depression | 57 | 11,192 | 55 | 47 |
| eating_disorder | 3 | 79 | 1,334 | 5 |
| schizophrenia | 27 | 133 | 9 | 617 |

**MentalRoBERTa**

| true ↓ / pred → | bipolar | depression | eating_disorder | schizophrenia |
|---|---|---|---|---|
| bipolar | 347 | 107 | 6 | 21 |
| depression | 51 | 11,197 | 49 | 54 |
| eating_disorder | 3 | 55 | 1,358 | 5 |
| schizophrenia | 27 | 128 | 8 | 623 |

Depression remains the dominant confusion sink under both encoders
(minority → depression: 116 / 79 / 133 for BERT, 107 / 55 / 128 for RoBERTa).

## 6. Paired comparison

| Statistic | Value |
|---|---|
| Rows where the two encoders predict the same label | 97.72% |
| Correct under both / neither | 13,365 / 394 |
| Correct under BERT only / RoBERTa only | 120 / 160 |
| McNemar exact test on per-row correctness | p = 0.0196 |
| Δ macro-F1, row-level paired bootstrap (2,000 resamples) | +0.0060, 95% CI [−0.0021, +0.0139] |
| Δ macro-F1, author-cluster paired bootstrap (1,000 resamples, 12,469 authors) | 95% CI [−0.0015, +0.0137] |

The two tests answer different questions and disagree at the 5% level: per-row
correctness (dominated by depression) favours MentalRoBERTa; macro-F1 (dominated by the two
rarest conditions) does not separate from zero.

## 7. Probability quality (relevant to C2, which consumes the softmax)

| | MentalBERT | MentalRoBERTa |
|---|---|---|
| Test NLL | 0.1904 | 0.1896 |
| ECE (15 equal-width bins, top-label) | 0.0311 | 0.0299 |
| Mean top-label confidence | 0.9915 | 0.9932 |

## 8. Caveats

- **One seed per encoder.** The bootstrap intervals cover test-set sampling, not training
  variance. The 1-epoch vs 3-epoch comparison in `baseline-results.md` §5 (Δ 0.0014) is a
  configuration change, not a seed-variance estimate, and is not used as one here.
- **No numeric threshold was set before the run.** The notebook's rule was qualitative
  ("a small gap supports the precedent argument; a large one is a limitation to state").
  The "small" classification in D-045 is therefore made after seeing the number. See D-045.
- **Proxy labels and in-distribution test**, as for C0.
- **Same pretraining corpus as MentalBERT** (Ji et al., 2022), so this run says nothing about
  the D-044 corpus-overlap question.
