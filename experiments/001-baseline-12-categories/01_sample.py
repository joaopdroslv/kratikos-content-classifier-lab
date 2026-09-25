"""Step 1 — draw the news-article sample.

Only articles that were ingested successfully are eligible: those are the ones with a vector in
the `news_articles` Qdrant collection (83.7k of 117.5k on dev; older articles predate
NEWS_ARTICLE_INGESTION_MIN_DATE).

Two strata, recorded in `stratum`:
- `uniform`: a plain random sample. Every unbiased number (the legacy category's accuracy, the
  share of articles no category fits) is computed on this stratum alone.
- `rare`: extra articles drawn per rare SOURCE category. The source label is noisy, but drawing on
  it still raises the count of the rare true classes, which the student needs to learn them.

Each row carries its topic id so that step 4 can split by topic: articles of the same story are
near-duplicates, and letting them straddle train/test would inflate every score.

    uv run python experiments/001-baseline-12-categories/01_sample.py
"""

import pandas as pd
from sqlalchemy import text

from lab.config import DATA_DIR, get_engine
from lab.taxonomy import CATEGORIES

OUT = DATA_DIR / "001" / "sample.parquet"
UNIFORM_SIZE = 2_400
RARE_PER_CATEGORY = 75
RARE_SOURCE_CATEGORIES = tuple(
    c.name for c in CATEGORIES if c.slug not in {"politics", "economy", "sports"}
)
CONTENT_CHARS = 1_500
SEED = 0.42  # setseed() takes a value in [-1, 1]

_SELECT = f"""
    SELECT n.id::text AS id, n.title, n.description,
           left(n.content, {CONTENT_CHARS}) AS content,
           n.category AS source_category, n.source_name, n.language, n.published_at,
           m.topic_id::text AS topic_id
    FROM public.news_articles n
    JOIN ai.ingested_news_articles i ON i.content_id = n.id AND i.successful IS TRUE
    LEFT JOIN ai.topic_members m
           ON m.content_id = n.id AND m.content_type = 'news_article'
"""


def main() -> None:

    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("SELECT setseed(:seed)"), {"seed": SEED})
        uniform = pd.read_sql(
            text(f"{_SELECT} ORDER BY random() LIMIT :n"),
            conn,
            params={"n": UNIFORM_SIZE},
        )
        uniform["stratum"] = "uniform"

        rare = pd.read_sql(
            text(f"""
                SELECT * FROM (
                    SELECT s.*, row_number() OVER (
                        PARTITION BY s.source_category ORDER BY random()
                    ) AS rn
                    FROM ({_SELECT}) s
                    WHERE s.source_category = ANY(:names)
                      AND NOT (s.id = ANY(:taken))
                ) r WHERE r.rn <= :per
                """),
            conn,
            params={
                "names": list(RARE_SOURCE_CATEGORIES),
                "taken": uniform["id"].tolist(),
                "per": RARE_PER_CATEGORY,
            },
        ).drop(columns="rn")
        rare["stratum"] = "rare"

    sample = pd.concat([uniform, rare], ignore_index=True)
    # An article with no topic row gets a group of its own for the split.
    sample["split_group"] = sample["topic_id"].fillna("article:" + sample["id"])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(OUT, index=False)

    print(f"wrote {len(sample)} articles to {OUT}")
    print(sample.groupby("stratum").size().to_string())
    print(sample["source_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
