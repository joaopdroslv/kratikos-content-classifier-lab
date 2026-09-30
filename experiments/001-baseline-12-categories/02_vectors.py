"""Step 2 — fetch each sampled article's vector from Qdrant.

The `news_articles` collection holds one flat (unnamed) 3072-d vector per point, and the point id
IS the article's Postgres id (`lab.vectors`). Articles whose point is missing are reported and
dropped.

    uv run python experiments/001-baseline-12-categories/02_vectors.py
"""

import numpy as np
import pandas as pd

from lab import vectors
from lab.config import DATA_DIR

SAMPLE = DATA_DIR / "001" / "sample.parquet"
OUT = DATA_DIR / "001" / "vectors.npz"


def main() -> None:

    ids = pd.read_parquet(SAMPLE)["id"].tolist()
    kept, X = vectors.fetch(ids)
    np.savez_compressed(OUT, ids=np.array(kept), vectors=X)
    print(
        f"wrote {len(kept)} vectors to {OUT}; {len(ids) - len(kept)} sampled ids had no point"
    )


if __name__ == "__main__":
    main()
