# MentalRoBERTa replication, explained from scratch

_A plain-English walkthrough of the encoder replication. Companion to the factual record in
[`roberta-replication-results.md`](roberta-replication-results.md). Assumes the vocabulary
introduced in [`baseline-results-explained.md`](baseline-results-explained.md) (precision,
recall, F1, macro-F1, confusion matrix)._

---

## 1. Why this run exists

The baseline (C0) uses **MentalBERT**. The same paper that released MentalBERT (Ji et al.,
2022) also released **MentalRoBERTa**, trained on the same mental-health Reddit text, and
that paper's own tables show MentalRoBERTa slightly ahead on most datasets. On SWMH, the
dataset most like this project's, the gap was about one F1 point (72.16 against 71.11).

So an examiner can fairly ask: *why build everything on the weaker of the two?* Until this
run the only answer was "most prior work uses MentalBERT", which is precedent, not evidence.
This run gives the evidence: train MentalRoBERTa in exactly the same way, on exactly the same
posts, and see how far apart the two land.

## 2. What "exactly the same way" means, and how it was checked

Everything except the encoder is held fixed: same training/validation/test posts, same
number of passes over the data, same batch size, same learning rate, same random seed, same
loss. Changing two things at once would make it impossible to say which one caused a
difference.

The split is rebuilt from the seed every time training runs, so "same posts" had to be
checked, not assumed. Each split file was turned into a fingerprint (a short hash of its
contents). All three fingerprints match, so both models were trained and tested on the same
posts, row for row.

## 3. What came out

| | MentalBERT | MentalRoBERTa |
|---|---|---|
| Macro-F1 (headline) | 0.8782 | 0.8841 |
| Accuracy | 0.9605 | 0.9634 |

MentalRoBERTa is ahead by **0.006 macro-F1**, a little over half a point. That is the same
direction as Ji et al.'s result and a little smaller than their one-point gap.

Condition by condition, it is ahead on depression, eating disorder and bipolar, and level on
schizophrenia. The ordering of conditions from easiest to hardest is identical under both
encoders, and both send most of their mistakes into "depression".

## 4. Is half a point real?

Two ways of asking, with two different answers.

- **"Across all 14,039 posts, does one model get more of them right?"** Each post is either
  right under both, wrong under both, or right under only one. There are 160 posts only
  MentalRoBERTa gets right and 120 only MentalBERT gets right. A McNemar test asks whether a
  160-to-120 split is likely by chance: p = 0.02, so by that measure MentalRoBERTa is genuinely
  a little better. This measure is dominated by depression, which is 81% of the test set.
- **"Is the macro-F1 gap bigger than the test set's own wobble?"** A bootstrap redraws the
  test set many times and recomputes the gap each time. The gap ranges from about −0.002 to
  +0.014, so the test set cannot rule out zero. This measure weights every condition
  equally, so it is driven by bipolar and schizophrenia, which have only 481 and 786 test
  posts and therefore wobble a lot.

Neither test captures a third source of wobble: retraining the same model with a different
random seed. Each encoder was trained once. A different seed could move either number by an
amount this run cannot measure.

## 5. What was decided (D-045)

The gap is small, in the direction Ji et al. predict, and inside the resampling uncertainty
for the headline metric. MentalBERT stays the encoder of record. MentalRoBERTa is reported
as a robustness line underneath the baseline table.

One point of honesty about "small": no number was fixed in advance for where "small" ends
and "large" begins. The judgement was made after seeing 0.006. The two reference points that
did exist before the run, Ji et al.'s one-point gap and a bootstrap interval that includes
zero, both put 0.006 on the small side, but a reader should weigh the call knowing it was
made afterwards.

## 6. What this does and does not show

**Does show:** the C0 headline is not a quirk of the chosen encoder. Swapping encoders moves
it by about half a point, and the structure the rest of the project leans on (which
conditions are hard, where mistakes go, how confident the model is) is the same.

**Does not show:**

- that MentalBERT is as good as MentalRoBERTa; on this data it is marginally worse;
- anything about the pretraining-overlap concern in D-044, because both encoders were
  pretrained on the same Reddit text;
- anything about clinical performance; both are tested on Reddit posts with subreddit labels,
  which are proxy labels for screening research, not diagnoses.
