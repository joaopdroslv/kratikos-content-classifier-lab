# kratikos-content-classifier-lab

Experiments for classifying Kratikos **news articles and posts** into **categories and
subcategories** (multi-label). The winning approach is later ported into `kratikos-ai-backend` as an
ingestion phase; this repository is where it is chosen and measured.

## Why this exists

A dev-database audit (2026-09-25) showed the current categorization is not usable for
preference-based suggestions:

- News categories come from the news source and are wrong for roughly half of an eyeballed sample.
  `Política` is a catch-all (47% of 117k articles).
- `Geral`, a name with no `public.categories` row, is ~8% of the corpus (mostly BBC, since 2026-08).
- Of the topics with 3 or more members, 3.5k out of 5.4k mix 2 to 8 categories: the same story is
  labelled differently.
- There are no subcategories.

## Hypothesis

Every item already has a `text-embedding-3-large` vector (3072-d) in Qdrant. A lightweight
multi-label head trained on those vectors (the **student**), with labels produced by an LLM (the
**teacher**), should classify as well as the teacher at no per-item API cost. This is the same
teacher–student setup behind the off-the-shelf IPTC classifier (Kuzman & Ljubešić, 2025), but it
reuses our own embeddings and our own taxonomy.

## Candidates compared in each experiment

| # | Approach | Training | Per-item cost |
|---|---|---|---|
| A | `classla/multilingual-IPTC-news-topic-classifier` mapped to our categories | none | local XLM-R large |
| B | Zero-shot: cosine between item vector and label-description vector | none | none |
| C | One-vs-rest logistic regression on the Qdrant vectors | LLM labels | none |

All candidates are scored against the same held-out LLM-labelled set, plus a human-reviewed golden
set that is **never** used for training.

## Setup

```bash
uv sync                      # core + dev
uv sync --group iptc         # only for candidate A (torch + ~2 GB checkpoint)
cp .env.example .env         # fill with the DEV values of kratikos-ai-backend — never production
uv run python -m lab.check   # verifies read-only access to Postgres and Qdrant
```

Format: `uv run isort src experiments && uv run black src experiments`.

## Layout

```
src/lab/            shared helpers (config, connections, loaders, metrics)
experiments/NNN-*/  one folder per experiment: README (question, method, result) + scripts
notebooks/          throwaway exploration
data/               pulled samples, vectors, labels (gitignored: production content)
models/             trained artifacts (gitignored)
```

## Rules

- **Read only.** Every Postgres transaction is opened `READ ONLY` (`src/lab/config.py`). The lab
  never writes to the database or to Qdrant.
- **Dev only.** `.env` points at the dev environment. Production is never a data source.
- **Data stays local.** `data/` and `models/` hold real user and news content and are never
  committed.
- **Code and docs in English, user-facing labels in pt-BR** (category names are shown to users),
  matching `kratikos-ai-backend/docs/standards/language.md`.
