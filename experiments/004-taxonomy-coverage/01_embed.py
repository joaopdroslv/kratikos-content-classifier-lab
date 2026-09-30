"""Step 1 — embed the taxonomy, and fetch the vectors of 002's validation.

Each subcategory becomes one prototype vector: `text-embedding-3-large` (the model behind every
Qdrant vector, so prototypes and articles share one space) of "master: subcategory. definition".
A master without subcategories (`lotteries`) stands in as its own subcategory. Each master also
gets a prototype of its own definition (`kind` = "master").

The 1,390 articles of 002's validation (step 7) test the gap scores in step 2: the teacher said,
for each one, whether a subcategory fits. Their vectors come from Qdrant `news_articles`.

Output (`data/004/`): `prototypes.parquet`, `validation_vectors.npz`.

    uv run python experiments/004-taxonomy-coverage/01_embed.py
"""

import numpy as np
import pandas as pd
from openai import OpenAI

from lab import metadata
from lab.config import DATA_DIR, get_qdrant
from lab.taxonomy_v2 import CATEGORIES, TAXONOMY_VERSION

DIR = DATA_DIR / "004"
VALIDATION_SAMPLE = DATA_DIR / "002" / "validation_sample.parquet"
MODEL = "text-embedding-3-large"
COLLECTION = "news_articles"
BATCH = 256


def prototype_texts() -> list[tuple[str, str, str]]:
    """(kind, key, text to embed): every subcategory (`master.subcategory`, or the bare master
    when it has none), then every master."""

    rows = []
    for master in CATEGORIES:
        if not master.facets:
            rows.append(
                ("subcategory", master.slug, f"{master.name}. {master.definition}")
            )
        for _, facet in master.facets:
            for sub in facet.subcategories:
                rows.append(
                    (
                        "subcategory",
                        f"{master.slug}.{sub.slug}",
                        f"{master.name}: {sub.name}. {sub.definition}",
                    )
                )
    rows += [("master", m.slug, f"{m.name}. {m.definition}") for m in CATEGORIES]
    return rows


def embed_prototypes() -> None:

    rows = prototype_texts()
    response = OpenAI().embeddings.create(
        model=MODEL, input=[text for _, _, text in rows]
    )
    pd.DataFrame(
        {
            "kind": [kind for kind, _, _ in rows],
            "key": [key for _, key, _ in rows],
            "text": [text for _, _, text in rows],
            "vector": [d.embedding for d in response.data],
            "model": MODEL,
            "taxonomy_version": TAXONOMY_VERSION,
            "created_at": metadata.now(),
        }
    ).to_parquet(DIR / "prototypes.parquet", index=False)
    print(f"embedded {len(rows)} prototypes (taxonomy {TAXONOMY_VERSION})")


def fetch_validation_vectors() -> None:

    ids = pd.read_parquet(VALIDATION_SAMPLE)["id"].tolist()
    qdrant = get_qdrant()
    found: dict[str, list[float]] = {}
    for start in range(0, len(ids), BATCH):
        points = qdrant.retrieve(
            COLLECTION,
            ids=ids[start : start + BATCH],
            with_vectors=True,
            with_payload=False,
        )
        found.update((str(p.id), p.vector) for p in points)
    kept = [id for id in ids if id in found]
    np.savez_compressed(
        DIR / "validation_vectors.npz",
        ids=np.array(kept),
        vectors=np.array([found[id] for id in kept], dtype=np.float32),
        sampled_at=np.array(metadata.now()),
    )
    print(f"fetched {len(kept)} validation vectors; {len(ids) - len(kept)} missing")


def main() -> None:

    DIR.mkdir(parents=True, exist_ok=True)
    embed_prototypes()
    fetch_validation_vectors()


if __name__ == "__main__":
    main()
