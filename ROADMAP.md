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
| — | Taxonomy proposal: master categories + subcategories | next |
| — | Team decision on the taxonomy | blocked on the proposal |
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

## Next

### 1. Taxonomy proposal (data for the team)

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
