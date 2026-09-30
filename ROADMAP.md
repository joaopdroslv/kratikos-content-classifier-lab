# Roadmap

What this lab is for, what is done, and what comes next. [README.md](README.md) holds the
hypothesis and the rules; each experiment's README holds its own method and result. This file is
the plan that ties them together: update it when an item closes or a decision is taken.

## Goals

1. **Classify news articles** into categories and subcategories (multi-label), well enough to drive
   preference-based suggestions, at no per-item API cost.
2. **Settle the taxonomy**: expand the master categories where the current 12 are too coarse, and
   define the subcategories under each one. Every proposal is backed by corpus numbers, and the
   final list is a product decision taken with the team.
3. **Hand over a winner** to `kratikos-ai-backend` as an ingestion phase: the model, its thresholds,
   the label definitions and the measured quality.

## Status

| # | Item | Status |
|---|---|---|
| 001 | Baseline on the 12 current categories (teacher–student on the Qdrant vectors) | **done** |
| 001 | Human review of the teacher (150 articles) | **done** |
| 002 | Taxonomy proposal: master categories + subcategories | **done**; 2.1 validated by the team (2026-09-30) |
| 003 | Teacher model: `gpt-6-luna`, `gpt-5.4-mini` vs `gpt-4.1-mini` on the human review | **done**: keep `gpt-4.1-mini` |
| — | Team decision on the taxonomy | **next** (input: [002 proposal](experiments/002-taxonomy-proposal/proposal.pt-BR.md)) |
| — | Labelling and student on the new taxonomy | blocked on the decision |
| — | Port to `kratikos-ai-backend` | later |
| — | Posts | on hold, see [Out of scope](#out-of-scope-for-now) |

## Done

**001 — baseline on the 12 categories** ([README](experiments/001-baseline-12-categories/README.md))

- Student (logistic regression on `text-embedding-3-large` vectors) vs human: **83%** primary
  accuracy; teacher (`gpt-4.1-mini`) vs human: **96%**; the source's category today: **28%**.
- Rare categories (education, housing, transport) are the weak spot; more labels help them.
- Secondaries are noisier than primaries: the teacher adds spurious ones (mostly public_safety,
  environment).

**002 — taxonomy proposal** ([README](experiments/002-taxonomy-proposal/README.md),
[proposal](experiments/002-taxonomy-proposal/proposal.pt-BR.md))

- Masters: clustered all 85.6k dev articles (via their 46k topics) inside each category.
  Geopolitics is ~10% of all articles (39% of today's politics); entertainment ~4%; lotteries
  1.3%. Adopted as new masters: `Geopolítica`, `Entretenimento`, `Loterias`, `Estilo de Vida`.
- Subcategories: induced from LLM tags on 3,825 articles, consolidated per master, then curated
  by hand into [`src/lab/taxonomy_v2.py`](src/lab/taxonomy_v2.py): **16 masters, 116
  subcategories**, multi-label, in a primary facet (sports by modality, politics by subject) and
  an optional cross-cutting facet. No geography: location is a separate filter.

## Next

### 1. Taxonomy proposal (data for the team) — done in 002, kept for the record

The 12 categories were used as-is in 001 on purpose, so 001 says little about what is missing: the
definitions in `src/lab/taxonomy.py` are broad enough to absorb almost everything (1.2% "no
category fits"). The proposal has to look for the gaps directly.

- **Master categories.** Candidates to split out or add, each measured before deciding:
  - `Geopolítica` out of `Política`: today's `politics` definition explicitly includes diplomacy,
    international relations and wars between states. Measure the share of `politics` that is
    international vs domestic (re-label the politics items, or cluster them by vector).
  - Others to check: entertainment/celebrities (today inside `Cultura`), science (inside
    `Tecnologia`), religion, weather, lifestyle/leisure, lotteries.
  - Cross-check against the 17 top-level IPTC Media Topics: what has no home in ours.
- **Subcategories.** Cluster the topic subjects (`ai.topic_members` → topic `subject`) within each
  category to see which subcategories carry real volume (e.g. Esportes → futebol, F1, tênis).
  Vectors only, no LLM cost. IPTC second-level topics as naming reference.
- **Output:** a pt-BR proposal with, per category and subcategory, a definition, its share of the
  corpus and example articles.

Masters and subcategories are decided together: whether something is a new master or a
subcategory of an existing one is the same question.

### 2. Labelling and student on the new taxonomy

A **new experiment** that re-runs the 001 pipeline on the chosen taxonomy. 001 itself stays as it
is: it answered its own question (can a student on our vectors track the teacher, and how bad is
the source's category), and that answer does not depend on the taxonomy. The 12 categories were a
deliberate choice so that 001 would not wait on a taxonomy decision, not an oversight to correct.
Editing 001 would erase the record of what was measured.

What carries over from 001:

| From 001 | Reused? |
|---|---|
| Sample (`sample.parquet`) and vectors (`vectors.npz`) | **Yes.** Same articles, no Qdrant cost, and a fair 12-vs-new comparison |
| Pipeline (sample → vectors → label → evaluate → review) | **Yes.** Same method |
| Teacher labels | **No.** Re-label on the new taxonomy (~3k `gpt-4.1-mini` calls; 003 found `gpt-6-luna` and `gpt-5.4-mini` 6 points worse against the human review) |
| Human review | **Partly.** Valid where a category did not change; anything on a new boundary (e.g. `Política` vs `Geopolítica`) is reviewed again |
| Per-category numbers and learning curve | **No.** Specific to the 12 |

The question this experiment answers: **with more categories and finer boundaries, does the
student still track the teacher?** `Política` vs `Geopolítica` is a harder boundary than
`Política` vs `Esportes`, and accuracy may drop.

Two practical changes when building it:

- **Do not overwrite `src/lab/taxonomy.py`**, which defines 001's labels; the new taxonomy is added
  next to it, versioned, so 001 stays reproducible.
- ~~Move the shared logic out of `04_evaluate.py`~~ **done** (2026-09-29): `lab.metrics` (score,
  per-label) and `lab.student` (label matrix, topic-grouped split, thresholds, student selection),
  taxonomy-agnostic (every function takes the ordered `slugs`). 001's outputs reproduce exactly.

On top of the 001 method:

- Teacher prompt v2: stricter secondaries (the 001 review's main finding).
- Hierarchical labels: category + subcategory; decide whether the student is one flat head or one
  head per level.
- Reinforce rare classes with student-guided sampling (label where the current model predicts a
  rare class) instead of uniform draws.
- Human review sheet on the new taxonomy, scored like `06_score_review.py`.

### 3. Topic-level category

Aggregate member articles' predictions into a category for each topic, and measure how stable it
is. Needed by the backend regardless, and it is the route for AI polls if the team chooses
inheritance (below).

### 4. Port to `kratikos-ai-backend`

Model artifact, thresholds and definitions as an ingestion phase; backfill of existing articles.

## Open decisions (for the team)

- **Taxonomy:** which master categories to add or split (e.g. `Geopolítica`), and the
  subcategory list. Input: the proposal from Next §1.
- **AI polls:** inherit the source topic's category? A gate already blocks polls that disagree
  semantically with the topic centroid. Caveat: on dev, 2,481 of 6,295 AI polls have no
  `poll_gap_verdicts` row via `candidate_id`, so their source topic is unknown.
- **User posts:** the author picks the category. Should a classifier do anything there (at most
  suggest a category while posting)? Only 84 user posts on dev, too few to evaluate.

## Out of scope for now

- **Posts** (AI polls and user posts): on hold until the team decides the two points above.
- **Candidate A** (off-the-shelf IPTC classifier): low priority, the student already clears the
  bar; may still serve as a reference when mapping the taxonomy.
