"""Article vectors from Qdrant, by id.

The `news_articles` collection holds one flat (unnamed) 3072-d `text-embedding-3-large` vector per
point, and the point id IS the article's Postgres id. Read only.
"""

from collections.abc import Sequence

import numpy as np
from tqdm import tqdm

from lab.config import get_qdrant

COLLECTION = "news_articles"
BATCH = 256


def fetch(ids: Sequence[str]) -> tuple[list[str], np.ndarray]:
    """The ids that have a point, in the given order, and their vectors; missing ids are dropped
    (compare the lengths to report them)."""

    qdrant = get_qdrant()
    found: dict[str, list[float]] = {}
    for start in tqdm(range(0, len(ids), BATCH), desc="qdrant"):
        points = qdrant.retrieve(
            COLLECTION,
            ids=list(ids[start : start + BATCH]),
            with_vectors=True,
            with_payload=False,
        )
        found.update((str(p.id), p.vector) for p in points)
    kept = [id for id in ids if id in found]
    return kept, np.array([found[id] for id in kept], dtype=np.float32)
