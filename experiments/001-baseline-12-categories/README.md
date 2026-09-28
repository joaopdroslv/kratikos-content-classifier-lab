# 001 — baseline on the 12 current categories

## Question

Can a lightweight head on the article vectors we already have (the **student**) reproduce the
labels of an LLM (the **teacher**) well enough to replace it? And how bad is the legacy category
the news source assigns?

Posts, subcategories and the off-the-shelf IPTC model are out of scope here — 001 uses only the
12 categories that already exist, so it needs no taxonomy decision.

## Method

| Step | Script | Output (`data/001/`) |
|---|---|---|
| 1 | `01_sample.py` | `sample.parquet`: 2,400 uniform + 675 rare-oversampled ingested news articles |
| 2 | `02_vectors.py` | `vectors.npz`: their `text-embedding-3-large` vectors from Qdrant `news_articles` |
| 3 | `03_label.py` | `labels.jsonl`: teacher `primary` + up to 2 `secondary` + `missing_category` |
| 4 | `04_evaluate.py` | `results.json`, `test_predictions.parquet` |
| 5 | `05_review_sheet.py` | `review.csv`: 150 uniform test articles for a human to judge the teacher |
| 6 | `06_score_review.py` | teacher, student and legacy primary accuracy against the filled `review.csv` |

- **Teacher:** `gpt-4.1-mini`, temperature 0, structured output over the 12 slugs, with the
  definitions in `src/lab/taxonomy.py` (prompt `001-v1`). Input: title + description + the first
  1,500 characters of content.
- **Split:** 75/25, **grouped by topic**. Articles on the same story are near-duplicates, and letting
  them fall on both sides would inflate every score.
- **Candidates:** `legacy` (the source's category), `zero_shot` (cosine against the vectors of the
  category definitions) and `student` (one-vs-rest logistic regression). C, the per-label thresholds
  and the no-category cut-off are tuned on out-of-fold training predictions, never on the test split.
- **Unbiased numbers** (legacy accuracy, the rate at which no category fits) come from the
  `uniform` stratum only.

Run order, from the repository root:

```bash
uv run python experiments/001-baseline-12-categories/01_sample.py
uv run python experiments/001-baseline-12-categories/02_vectors.py
uv run python experiments/001-baseline-12-categories/03_label.py
uv run python experiments/001-baseline-12-categories/04_evaluate.py
uv run python experiments/001-baseline-12-categories/05_review_sheet.py
# fill review.csv, then
uv run python experiments/001-baseline-12-categories/06_score_review.py
```

## Result (2026-09-25, dev)

3,075 articles labelled by the teacher, no failures. Split: train 2,321 / test 754, grouped by topic.

**Candidates vs teacher, test split** (the `uniform` column is the unbiased stratum):

| Candidate | primary_acc | uniform | primary_in_set | micro_f1 | macro_f1 |
|---|---|---|---|---|---|
| legacy | 0.369 | 0.327 | 0.507 | 0.370 | 0.375 |
| zero_shot | 0.614 | 0.595 | 0.762 | 0.654 | 0.621 |
| **student** | **0.829** | **0.839** | **0.968** | **0.848** | **0.802** |

- **The student tracks the teacher.** Its primary is one of the teacher's labels 97% of the time.
  Most of the remaining primary disagreement is a primary/secondary swap, which does not matter
  for ranking by preference.
- **The legacy category agrees with the teacher on about 1 article in 3.** Per source label
  (uniform stratum): `Política` 29% (22% is actually sport, mostly the BBC feed), `Economia` 29%,
  `Tecnologia` 22%, `Esportes` 86%. `Geral` goes 33% to sports and 24% to politics.
- **Zero-shot against the category definitions is not enough** (0.60). A trained head is needed.
- **Weakest labels:** education (F1 0.62, support 21), housing (0.67, 30), transport (0.70),
  human_rights (0.75). These are the rare classes.
- **Learning curve** (default thresholds): primary_acc plateaus from ~500 labels (0.81), while
  macro_f1 keeps climbing (0.55 → 0.64 → 0.72 → 0.78 at 250/500/1k/2.3k). More labels help the
  rare classes, not the common ones.
- **Real distribution** (teacher, uniform): sports 26%, politics 20%, economy 13%,
  public_safety 10%, culture 7%, technology 6%, health 5%. The teacher found no fitting category
  for only 1.2%. The recurring gap was lotteries. The definitions are broad enough to absorb
  celebrity news into culture, so this is a weak taxonomy signal by construction.

**Human review (2026-09-28).** The 150 uniform test articles in `review.csv` were judged by a
person. Primary accuracy against that judgement (95% Wilson interval):

| Candidate | primary_acc | 95% CI |
|---|---|---|
| **teacher** | **0.960** (144/150) | 0.915 – 0.982 |
| student | 0.833 (125/150) | 0.766 – 0.884 |
| legacy | 0.280 (42/150) | 0.214 – 0.357 |

- **The teacher is a sound ceiling.** Its 6 misses are boundary cases: a lottery result (no
  category fits), celebrity/TV-network quarrels labelled politics instead of culture, a student
  protest movement labelled public_safety instead of politics.
- **Student vs human (0.83) ≈ student vs teacher (0.84)**, so agreement with the teacher is a fair
  proxy for correctness at this teacher quality.
- **Secondaries are noisier than primaries:** 14 of the 150 notes flag a wrong or forced secondary
  (mostly a spurious public_safety or environment). Worth a prompt tweak before labelling more.
- 14 rows came back from the spreadsheet with the teacher's columns blank and were judged blind
  (teacher 12/14 on those). `06_score_review.py` joins by id and reads only the reviewer's
  columns, so the export loss does not affect the numbers.

**Conclusion:** the teacher–student approach on existing vectors works for the 12 categories.
The student classifies with no per-item API cost, and with about 3× the legacy agreement.
