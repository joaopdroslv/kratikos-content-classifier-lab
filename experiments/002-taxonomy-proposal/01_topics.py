"""Step 1 — pull every topic: its subject, size, scope, and centroid vector.

A topic groups the news articles of one story; its `centroid` vector (Qdrant `topics`) is the
mean of its members' `text-embedding-3-large` vectors, so it lives in the same space as the
article vectors 001 trained on. Most topics have a single member, so the topics cover the whole
ingested corpus, and each carries a readable `subject` to name clusters with.

    uv run python experiments/002-taxonomy-proposal/01_topics.py
"""

import numpy as np
import pandas as pd
from sqlalchemy import text
from tqdm import tqdm

from lab.config import DATA_DIR, get_engine, get_qdrant

OUT_DIR = DATA_DIR / "002"
COLLECTION = "topics"
VECTOR = "centroid"
BATCH = 512


def main() -> None:

    with get_engine().connect() as conn:
        topics = pd.read_sql(
            text("""
                SELECT
                    t.id::text AS id,
                    t.subject,
                    t.scope,
                    t.location_country,
                    count(tm.content_id) AS members
                FROM ai.topics AS t
                JOIN ai.topic_members AS tm ON TRUE
                    AND tm.topic_id = t.id
                    AND tm.content_type = 'news_article'
                GROUP BY t.id
                """),
            conn,
        )

    qdrant = get_qdrant()
    found: dict[str, list[float]] = {}
    offset = None
    with tqdm(total=qdrant.count(COLLECTION).count, desc="qdrant") as bar:
        while True:
            points, offset = qdrant.scroll(
                COLLECTION,
                limit=BATCH,
                offset=offset,
                with_payload=False,
                with_vectors=[VECTOR],
            )
            for point in points:
                found[str(point.id)] = point.vector[VECTOR]
            bar.update(len(points))
            if offset is None:
                break

    topics = topics[topics["id"].isin(found)].reset_index(drop=True)
    X = np.array([found[id] for id in topics["id"]], dtype=np.float32)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    topics.to_parquet(OUT_DIR / "topics.parquet", index=False)
    np.save(OUT_DIR / "centroids.npy", X)
    print(
        f"wrote {len(topics)} topics ({int(topics['members'].sum())} articles);"
        f" {len(found) - len(topics)} Qdrant points had no news members"
    )


if __name__ == "__main__":
    main()
