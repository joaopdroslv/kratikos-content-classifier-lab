# 004 — taxonomy coverage: does 2.1 miss recurring subjects?

**Question.** The 2.1 masters came from the whole corpus (002 part 1), but its subcategories were
induced from 3,825 tagged articles (002 part 2). Does the corpus hold recurring subjects that no
subcategory covers, and how large are they? A gap found here becomes taxonomy 2.2 before the
training set is labelled.

**Prior evidence (002's validation).** On 1,390 unseen articles, the `missing_subcategory` names
the teacher proposed were almost all singletons: no gap above ~0.35% of articles is likely to
have gone unseen. Two limits: the same subject may be named in several ways, and the small
masters had only 28–68 validation articles each.

## Method

| Step | Script | Output (`data/004/`) |
|---|---|---|
| 1 | `01_embed.py` | `prototypes.parquet`: one `text-embedding-3-large` vector per subcategory and per master of 2.1; `validation_vectors.npz`: the Qdrant vectors of 002's validation articles |
| 2 | `02_detectors.py` | `detectors.md`: whether any distance to the prototypes separates covered from uncovered validation articles |
| 3 | `03_clusters.py` | `topics_clustered.parquet`, `clusters.md`: all 45,970 topics of 002 (every ingested article) in 400 k-means clusters |
| 4 | `04_judge.py` | `judge_<hash>.jsonl`, `missing_<hash>.jsonl`, `gaps.md`: every cluster judged against 2.1 by `gpt-4.1`; 002's missing names grouped into themes |

```bash
uv run python experiments/004-taxonomy-coverage/01_embed.py   # ... through 04_judge.py
```

- **The first design filtered the corpus by distance** (keep only the topics far from every
  prototype, judge those). Step 2 rules it out, see Result, so step 3 clusters everything and
  step 4 judges every cluster. With 400 clusters the average one is 0.25% of the articles: a
  subject large enough to be a subcategory gets a cluster of its own, or shows up as a
  cluster's `uncovered_minority`.
- **Topics stand in for articles**, as in 002: a topic's centroid is the mean of its members'
  vectors, and clusters are weighted by member count, so shares are shares of articles.
- **Dev corpus.** Dev leans on UK and Portuguese feeds (002's caveat). A gap that matters to a
  Brazilian audience but is rare in dev cannot be found here; that is left to the rejection
  threshold on the student once it runs on production data.

| Step | Determinism / resumability / live data |
|---|---|
| `01_embed` | **Reads Qdrant** (the 1,394 validation vectors; `sampled_at` in the npz). Embeddings cost cents; re-run after any taxonomy change |
| `02_detectors` | Deterministic, local |
| `03_clusters` | Deterministic (seeded), local; reads 002's `topics_categorised.parquet` and `centroids.npy` |
| `04_judge` | **Resumable, costs API** (~US$3 for 400 clusters with `gpt-4.1`); answers go to files named by the prompt's hash. `--limit N` judges the first N clusters |

## Result

### Distance does not detect gaps (step 2, 2026-09-30)

On 002's validation, 1,315 covered and 75 uncovered articles:

| score | AUC (0.5 = blind) |
|---|---|
| highest cosine to any subcategory | 0.46 |
| same, each prototype standardised over the corpus | 0.43 |
| best minus second-best subcategory | 0.48 |
| best subcategory of the nearest master minus that master | 0.69 |

The prototypes do know **which** subcategory an article belongs to (a teacher subcategory is among
the 3 nearest for 82% of covered articles), but not **whether** one fits: an article that no
subcategory of its master describes still sits as close to a sibling as a covered one does. The
best score, 0.69, is too weak to throw away most of the corpus unread. It is also a warning for
the planned rejection threshold on the student: a trained head is not a cosine to a definition,
but that it catches novel subjects has to be measured (hold a subcategory out), not assumed.

### Gaps (step 4, 2026-09-30, taxonomy 2.1)

366 of 400 clusters judged (92.3% of articles; 34 hit the `gpt-4.1` rate limit and were not
retried: the taxonomy is now 2.2, so a re-run hashes to a new file and judges all 400 again),
US$1.89. Verdicts by share of articles: existing 80.2%, new subcategory 11.4% (48
clusters), new master 0%, noise 0.7%. Report: `data/004/gaps.md`.

Read by the product owner and by Claude independently, with the same outcome. The 48 candidates
fall into a few kinds, none of which is a subcategory:

| Kind | Examples (cluster) |
|---|---|
| One competition or aspect inside a modality | World Cup (57), tournament draws (302), national squads (278, 227), injuries (5 clusters, ~0.7%), refereeing (174, 35) |
| A place or entity (the no-geography rule) | African development (338), China (222), ICE (210), one bank (314) |
| A story format, not a subject | obituaries (200, 164, 332), life stories (280, 386), promotions (316, 125), rankings (289, 303), fairs and events (398, 66) |
| A boundary between masters, already written | health tech (51), aviation incidents (15, accidents → public_safety) |

**The test that sorts them is the product's, not the corpus's: would a reader pick it as an
interest?** Nobody follows "injuries" or "obituaries". Of the candidates and the clusters'
uncovered minorities, two pass: **cryptocurrencies** (cluster 14, 0.29%, plus fintech
minorities) and **mental health**, which 2.1 already has (`health.mental_health`; the judge
missed it from a wellbeing cluster).

The judge also makes plain mistakes (Mega-Sena → `lifestyle.food_recipes`), and its prompt did not
carry the product test, so it names a specific theme in almost every cluster. For the next
taxonomy review: put "would a reader follow it?" and the facet axis in the judge's prompt.

**Conclusion.** 2.1 has no missing subcategory worth adding except cryptocurrencies:
`economy.cryptocurrencies` was added, **taxonomy 2.2** (16 masters, 114 subcategories). The limit
from the Method stands: this is the dev corpus; subjects rare in dev are left to monitoring in
production.
