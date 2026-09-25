"""Step 2 — fetch each sampled article's vector from Qdrant.

The `news_articles` collection holds one flat (unnamed) 3072-d vector per point, and the point id
IS the article's Postgres id. Articles whose point is missing are reported and dropped.

    uv run python experiments/001-baseline-12-categories/02_vectors.py
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from lab.config import DATA_DIR, get_qdrant

SAMPLE = DATA_DIR / "001" / "sample.parquet"
OUT = DATA_DIR / "001" / "vectors.npz"
COLLECTION = "news_articles"
BATCH = 256


def main() -> None:

    ids = pd.read_parquet(SAMPLE)["id"].tolist()
    qdrant = get_qdrant()

    found: dict[str, list[float]] = {}
    for start in tqdm(range(0, len(ids), BATCH), desc="qdrant"):
        points = qdrant.retrieve(
            COLLECTION,
            ids=ids[start : start + BATCH],
            with_vectors=True,
            with_payload=False,
        )
        for point in points:
            found[str(point.id)] = point.vector

    missing = [id for id in ids if id not in found]
    kept = [id for id in ids if id in found]
    np.savez_compressed(
        OUT,
        ids=np.array(kept),
        vectors=np.array([found[id] for id in kept], dtype=np.float32),
    )
    print(
        f"wrote {len(kept)} vectors to {OUT}; {len(missing)} sampled ids had no point"
    )


if __name__ == "__main__":
    main()
