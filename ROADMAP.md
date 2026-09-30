# Roadmap

What this lab is for, what is done, and what comes next. [README.md](README.md) holds the
hypothesis and the rules; each experiment's README holds its own method and result. This file is
the plan that ties them together: update it when an item closes or a decision is taken.

## Goals

1. **Classify news articles** into categories and subcategories (multi-label), well enough to drive
   preference-based suggestions, at no per-item API cost.
2. **Settle the taxonomy**: expand the master categories where the current 12 are too coarse, and
   define the subcategories under each one. Every proposal is backed by corpus numbers, and the
   final list is a product decision taken with the team. **Done: taxonomy 2.2.**
3. **Hand over a winner** to `kratikos-ai-backend` as an ingestion phase: the model, its thresholds,
   the label definitions and the measured quality.

## Status

| # | Item | Status |
|---|---|---|
| 001 | Baseline on the 12 current categories (teacher–student on the Qdrant vectors) + human review | **done** |
| 002 | Taxonomy proposal: master categories + subcategories | **done**; 2.1 validated by the team (2026-09-30) |
| 003 | Teacher model on the 12 categories: `gpt-6-luna`, `gpt-5.4-mini` vs `gpt-4.1-mini` | **done**: keep `gpt-4.1-mini` |
| 004 | Taxonomy coverage: recurring subjects 2.1 misses, over the whole corpus | **done**: crypto added → **2.2** |
| 005 | Teacher model on taxonomy 2.2 | **next** ([§1](#1-teacher-on-taxonomy-22)) |
| 006 | Labelling set, golden set and student on 2.2 | after 005 ([§2](#2-labelling-set-and-student-on-22)) |
| — | Topic-level category | later ([§3](#3-topic-level-category)) |
| — | Port to `kratikos-ai-backend` | later ([§4](#4-port-to-kratikos-ai-backend)) |
| — | Posts | on hold, see [Out of scope](#out-of-scope-for-now) |

Experiment numbers for 005 and 006 are provisional.

## Done

**001 — baseline on the 12 categories** ([README](experiments/001-baseline-12-categories/README.md))

- Student (logistic regression on `text-embedding-3-large` vectors) vs human: **83%** primary
  accuracy; teacher (`gpt-4.1-mini`) vs human: **96%**; the source's category today: **28%**.
- Rare categories (education, housing, transport) are the weak spot. The learning curve had not
  flattened: macro-F1 0.55 → 0.64 → 0.72 → 0.78 at 250 / 500 / 1k / 2.3k labels.
- Secondaries are noisier than primaries: the teacher adds spurious ones (mostly public_safety,
  environment).

**002 — taxonomy proposal** ([README](experiments/002-taxonomy-proposal/README.md),
[proposal](experiments/002-taxonomy-proposal/proposal.pt-BR.md))

- Masters: clustered all 85.6k dev articles (via their 46k topics) inside each category.
  Geopolitics is ~10% of all articles (39% of today's politics); entertainment ~4%; lotteries
  1.3%. Adopted as new masters: `Geopolítica`, `Entretenimento`, `Loterias`, `Estilo de Vida`.
- Subcategories: induced from LLM tags on 3,825 articles, consolidated per master, then curated
  by hand into [`src/lab/taxonomy_v2.py`](src/lab/taxonomy_v2.py), multi-label, in a primary
  facet (sports by modality, politics by subject) and an optional cross-cutting facet. No
  geography: location is a separate filter.
- Validated by `gpt-4.1` on 1,390 unseen articles (92–100% coverage per master), revised by hand
  to 2.1 (113 subcategories), approved by the team on 2026-09-30.

**003 — teacher model** ([README](experiments/003-teacher-model/README.md))

- On the 12 categories, `gpt-6-luna` and `gpt-5.4-mini` are ~6 points below `gpt-4.1-mini`
  against the human review. Not yet tested on taxonomy 2.x.

**004 — taxonomy coverage** ([README](experiments/004-taxonomy-coverage/README.md))

- All 85.6k dev articles in 400 clusters, each judged against 2.1 by `gpt-4.1`. The 48 "new
  subcategory" verdicts were almost all too specific (one competition inside a modality), tied
  to a place or entity, or a story format (obituaries, rankings, promotions). Only
  cryptocurrencies passed the product test ("would a reader follow it?"): taxonomy **2.2**,
  16 masters, 114 subcategories.
- Distance to the subcategory definitions does not detect uncovered articles (AUC ≤ 0.69), so
  the student's rejection threshold has to be measured, not assumed.

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-09-30 | **No trained "Geral" class.** The student gets a rejection threshold instead: below it, an article is "não classificado" | A catch-all class learns its own training examples, not what is new; a novel subject would land confidently in an existing class |
| 2026-09-30 | **The "não classificado" pool feeds the next revision**: labelled by the teacher, it splits into student errors (new training rows) and new subjects (candidates for the next taxonomy) | Keeps the taxonomy and the training set current without re-sampling at random |
| 2026-09-30 | **A larger training set than 001's**, sampled per subcategory, at a higher labelling cost | 114 subcategories on ~4k articles leaves the tail with 5–10 examples; 001's curve was still rising |
| 2026-09-30 | **Product test for a subcategory: would a reader follow it as an interest?** | 004: most corpus-driven candidates are formats, events or places, not interests |

## Next

### 1. Teacher on taxonomy 2.2

003 chose `gpt-4.1-mini` on the 12 categories, but the v2 prompt is ~7k tokens with facets and
optional fields, and in 002 `gpt-4.1-mini` overfilled an optional field (the aspect tag on 39 of 40
trial articles). Before labelling thousands of articles:

- Run `gpt-4.1-mini` on 002's 1,390 validation articles with the 2.2 prompt and compare with
  `gpt-4.1` (re-run on 2.2, or on 2.1's answers where 2.2 did not change): primary agreement,
  subcategory agreement, labels per article, and where they disagree. ~US$1.5–4.
- Output: which teacher labels the training set, and its cost per 1k articles.

### 2. Labelling set and student on 2.2

A **new experiment** that re-runs the 001 pipeline on taxonomy 2.2. 001 stays as it is: it
answered its own question, and editing it would erase the record of what was measured.

- **Sample per subcategory, not uniformly.** Target ≥100 positives per subcategory (≥50 for the
  rarest), roughly 7–10k articles. Candidates for small subcategories are found with the
  prototypes of 004 and a first student, the way 001 oversampled rare classes and 002's step 4
  filled the small masters. A uniform stratum is kept for unbiased shares.
- **In rounds** (e.g. 4k → 7k → 10k), with the per-subcategory macro-F1 measured after each; stop
  when it flattens.
- **Human golden set on 2.2**: 150–300 articles reviewed by the team, never used for training.
  More teacher labels do not fix systematic teacher errors (e.g. elections vs parties); only
  this set measures them.
- **Reused from 001:** the method (topic-grouped split, thresholds tuned out of fold, `lab.metrics`,
  `lab.student`) and its 3,075 sampled articles and vectors, re-labelled on 2.2. The teacher
  labels and per-category numbers do not carry over.
- **Hierarchy:** decide whether the student is one flat head over masters + subcategories or one
  head per level.
- **Rejection threshold:** measure it by holding a whole subcategory out of training and checking
  how much of it the student rejects, at what cost to the known classes. This is the number the
  "não classificado" state rests on.
- Teacher prompt: stricter secondaries (the 001 review's main finding).

The question: **with 16 masters and 114 subcategories, does the student still track the
teacher?** Finer boundaries (`Política` vs `Geopolítica`) may cost accuracy.

### 3. Topic-level category

Aggregate member articles' predictions into a category for each topic, and measure how stable it
is. Needed by the backend regardless, and it is the route for AI polls if the team chooses
inheritance (below).

### 4. Port to `kratikos-ai-backend`

Model artifact, thresholds and definitions as an ingestion phase; backfill of existing articles.
Once live, the "não classificado" pool is monitored on production data: dev leans on UK and
Portuguese feeds, and subjects rare in dev can only show up there.

## Open decisions (for the team)

- **AI polls:** inherit the source topic's category? A gate already blocks polls that disagree
  semantically with the topic centroid. Caveat: on dev, 2,481 of 6,295 AI polls have no
  `poll_gap_verdicts` row via `candidate_id`, so their source topic is unknown.
- **User posts:** the author picks the category. Should a classifier do anything there (at most
  suggest a category while posting)? Only 84 user posts on dev, too few to evaluate.

## Out of scope for now

- **Posts** (AI polls and user posts): on hold until the team decides the two points above.
- **Candidate A** (off-the-shelf IPTC classifier): low priority, the student already clears the
  bar; may still serve as a reference when mapping the taxonomy.
