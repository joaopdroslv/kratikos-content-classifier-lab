# 003 — teacher model: cheaper or newer alternatives to gpt-4.1-mini

**Question.** Can a newer OpenAI model replace `gpt-4.1-mini` (US$0.40 / 1.60 per 1M tokens in /
out) as the teacher that labels the training set, at equal or better quality? Candidates:
`gpt-6-luna` (released 2026-09-23, US$0.10 / 0.50) and `gpt-5.4-mini` (US$0.75 / 4.50).

## Method

| Step | Script | Output |
|---|---|---|
| 1 | `01_human_set.py` | `data/003/human_<candidate>.jsonl`, `human_results.json`: candidates on the 150 articles of the 001 human review, scored against the human primary |

- Same prompt and schema as 001's teacher (`001/03_label.py`, prompt_version `001-v1`, the 12
  categories), same article rendering. The reference is 001's own `gpt-4.1-mini` answers.
- Candidates: `gpt-6-luna` at `reasoning_effort` `none` (temperature 0) and `low` (the API only
  accepts the default temperature above `none`, so those answers are not deterministic);
  `gpt-5.4-mini` at `none`.
- Every answer records its token `usage`; `lab.pricing` turns it into dollars.
- **Bias:** the reviewer saw gpt-4.1-mini's answer and accepted it on most rows, so on ambiguous
  articles the truth leans toward it. The misses are printed to be read, not only counted.

```bash
uv run python experiments/003-teacher-model/01_human_set.py
```

| Step | Determinism / resumability / live data |
|---|---|
| `01_human_set` | Resumable (per-candidate JSONL). `none` is temperature 0; `low` is not deterministic. Does not read the live database (articles come from `data/001/sample.parquet`) |

## Result (2026-09-29)

**Primary vs human, 001 review (150 articles, 95% Wilson interval):**

| candidate | primary_acc | interval | cost of the 150 |
|---|---|---|---|
| **gpt-4.1-mini** (001) | **0.960** (144/150) | 0.915 – 0.982 | ~US$0.06 (estimated; 001 did not record usage) |
| gpt-6-luna, effort `none` | 0.900 (135/150) | 0.842 – 0.939 | US$0.016 |
| gpt-6-luna, effort `low` | 0.900 (135/150) | 0.842 – 0.939 | US$0.020 |
| gpt-5.4-mini, effort `none` | 0.900 (135/150) | 0.842 – 0.939 | US$0.121 |

- **Luna is worse, not better.** Where it disagrees with gpt-4.1-mini (18 articles at `none`), the
  human sides with gpt-4.1-mini 13 times and with Luna 4 times. Part of that gap is the reviewer's
  anchoring on gpt-4.1-mini, but it holds on the 3 blind rows too, and the direction is
  consistent: Luna files an article by its money or political angle instead of its subject (a
  PlayStation game as `economy`, a new car model as `economy`, an explosion as `transport`).
- **gpt-5.4-mini is no better, and costs about twice as much as gpt-4.1-mini.** Where it disagrees
  with gpt-4.1-mini (15 articles) the human sides with gpt-4.1-mini 11 times and with it 2 times.
  Its misses include a shark bite as `health` and an explosion as `transport`.
- **The two newer models do not share one blind spot.** They agree with each other on 89% of the
  articles; where both agree against gpt-4.1-mini (7), the human still sides with gpt-4.1-mini 4
  times. The anchoring of the review favours gpt-4.1-mini, but it does not explain a 6-point gap
  that two unrelated models reproduce with mostly different errors.
- **Reasoning does not help.** `low` spends ~52 reasoning tokens per article, costs 25% more, and
  lands at the same 90%, on a partly different set of misses.
- **The saving is small in dollars.** 001's prompt is ~930 input tokens. On the v2 prompt (~7k
  tokens, mostly cacheable), labelling ~5k articles would cost roughly US$1–4 with Luna and
  US$4–14 with gpt-4.1-mini. That is a few dollars saved for a teacher about 6 points less
  accurate, and every error in the teacher becomes an error the student learns.

**Conclusion.** Keep `gpt-4.1-mini` (or better) as the teacher; neither newer model beats it on
this task. Not run: agreement of the candidates with gpt-4.1 on the v2 taxonomy (the 002
validation sample); worth running only if a cheaper teacher is reconsidered.
