"""Step 4 — pull extra articles for the masters the 001 sample holds too few of.

The 001 sample gives the small masters a few dozen articles each (housing 31, education 66),
too few to induce subcategories from. Extra articles are drawn from the topics the 001 student put
under the matching v1 category (step 2): a noisy filter, but step 5 re-labels every article's
master anyway, so a wrong pick only costs a call.

`culture` covers both v2 `culture` and `entertainment`, which were one category in v1.

Output: `data/002/extra_sample.parquet`, same columns as the 001 sample, `stratum` = `extra:<v1>`.

    uv run python experiments/002-taxonomy-proposal/04_extra_sample.py
"""

import pandas as pd
from sqlalchemy import text

from lab.config import DATA_DIR, get_engine

DIR = DATA_DIR / "002"
EXTRA = {
    "housing": 150,
    "education": 120,
    "transport": 120,
    "human_rights": 100,
    "culture": 200,
    "environment": 60,
}
CONTENT_CHARS = 1_500
SEED = 0.42


def main() -> None:

    topics = pd.read_parquet(DIR / "topics_categorised.parquet")
    taken = pd.read_parquet(DATA_DIR / "001" / "sample.parquet")["id"].tolist()

    parts = []
    with get_engine().connect() as conn:
        conn.execute(text("SELECT setseed(:seed)"), {"seed": SEED})
        for category, n in EXTRA.items():
            topic_ids = topics.loc[topics["category"] == category, "id"].tolist()
            part = pd.read_sql(
                text(f"""
                    SELECT
                        na.id::text AS id,
                        na.title,
                        na.description,
                        LEFT(na.content, {CONTENT_CHARS}) AS content,
                        na.category AS source_category,
                        na.source_name,
                        na.language,
                        na.published_at,
                        tm.topic_id::text AS topic_id
                    FROM ai.topic_members AS tm
                    JOIN public.news_articles AS na ON na.id = tm.content_id
                    WHERE TRUE
                        AND tm.content_type = 'news_article'
                        AND tm.topic_id = ANY(CAST(:topics AS uuid[]))
                        AND NOT (na.id::text = ANY(:taken))
                    ORDER BY random()
                    LIMIT :n
                    """),
                conn,
                params={"topics": topic_ids, "taken": taken, "n": n},
            )
            part["stratum"] = f"extra:{category}"
            taken += part["id"].tolist()
            parts.append(part)

    sample = pd.concat(parts, ignore_index=True)
    sample["split_group"] = sample["topic_id"].fillna("article:" + sample["id"])
    sample.to_parquet(DIR / "extra_sample.parquet", index=False)
    print(f"wrote {len(sample)} extra articles")
    print(sample["stratum"].value_counts().to_string())


if __name__ == "__main__":
    main()
