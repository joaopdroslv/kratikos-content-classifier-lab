"""Step 3 — cut the whole corpus into fine clusters, each small enough to judge by reading.

Step 2 showed that no distance to the prototypes separates covered from uncovered articles, so
nothing is filtered out: every one of the 45,970 topics of 002 (all 85,586 ingested articles) is
clustered, and step 4 judges every cluster. k-means on the topic centroids, weighted by member
count (as 002's step 3), across all masters at once, since a gap may straddle two. With `K`
clusters, the average cluster is 1/K of the articles: a subject much larger than that gets a
cluster of its own, which is the size of gap worth a subcategory.

Each cluster is listed with its share of all articles, its scope mix, the subcategory prototypes
its topics are nearest to (a hint for the reader, right on the master ~3 times in 4), and the
subjects closest to its centre.

Output (`data/004/`): `topics_clustered.parquet`, `clusters.md`.

    uv run python experiments/004-taxonomy-coverage/03_clusters.py
"""

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from lab.config import DATA_DIR

DIR = DATA_DIR / "004"
DIR_002 = DATA_DIR / "002"
K = 400
PCA_DIMS = 256
EXAMPLES = 15
SEED = 0


def normalise(X: np.ndarray) -> np.ndarray:

    return X / np.linalg.norm(X, axis=1, keepdims=True)


def nearest_subcategory(X: np.ndarray) -> list[str]:

    prototypes = pd.read_parquet(DIR / "prototypes.parquet")
    part = prototypes[prototypes["kind"] == "subcategory"]
    keys = part["key"].tolist()
    best = (X @ normalise(np.stack(part["vector"])).T).argmax(axis=1)
    return [keys[i] for i in best]


def main() -> None:

    topics = pd.read_parquet(DIR_002 / "topics_categorised.parquet")
    X = normalise(np.load(DIR_002 / "centroids.npy"))
    topics["nearest"] = nearest_subcategory(X)

    Z = normalise(PCA(n_components=PCA_DIMS, random_state=SEED).fit_transform(X))
    model = KMeans(n_clusters=K, n_init=4, random_state=SEED).fit(
        Z, sample_weight=topics["members"].to_numpy()
    )
    topics["cluster"] = model.labels_
    topics["distance"] = np.linalg.norm(
        Z - model.cluster_centers_[model.labels_], axis=1
    )
    total = topics["members"].sum()

    clusters = topics.groupby("cluster").agg(
        articles=("members", "sum"),
        topics=("id", "size"),
        international=("scope", lambda s: (s == "internacional").mean()),
    )
    lines = [
        "# 004 — the corpus in fine clusters (raw evidence for step 4)\n",
        f"{len(topics):,} topics, {total:,} articles, {K} clusters.\n",
    ]
    for name, row in clusters.sort_values("articles", ascending=False).iterrows():
        members = topics[topics["cluster"] == name]
        nearest = Counter()
        for key, n in zip(members["nearest"], members["members"]):
            nearest[key] += n
        examples = members.sort_values(["distance", "members"], ascending=[True, False])
        lines.append(
            f"## {name} — {row['articles'] / total:.2%} of all articles; {row['topics']} topics;"
            f" international {row['international']:.0%}\n"
        )
        lines.append(
            "Nearest subcategories: "
            + ", ".join(
                f"{k} ({n / row['articles']:.0%})" for k, n in nearest.most_common(3)
            )
            + "\n"
        )
        lines.extend(f"- {s}" for s in examples["subject"].head(EXAMPLES))
        lines.append("")

    topics.to_parquet(DIR / "topics_clustered.parquet", index=False)
    (DIR / "clusters.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {K} clusters to {DIR / 'clusters.md'}")


if __name__ == "__main__":
    main()
