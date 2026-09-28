"""Step 3 — cluster the topics inside each category.

Within each predicted category, k-means on the topic centroids (weighted by member count, so a
cluster's size is its share of ARTICLES, not of topics). k grows with the category's size, on
purpose finer than a final subcategory list: clusters are merged by reading, not split.

A cluster is evidence for a **subcategory** when it is large and coherent, and for a **new master
category** when it is large and coherent but does not belong under its parent (e.g. geopolitics
under `politics`, celebrities under `culture`). The step only produces the evidence: every cluster
with its share, its scope mix and the subjects closest to its centre, in `clusters.md`.

    uv run python experiments/002-taxonomy-proposal/03_clusters.py
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from lab.config import DATA_DIR
from lab.taxonomy import BY_SLUG, SLUGS

DIR = DATA_DIR / "002"
TOPICS_PER_CLUSTER = 400
K_MIN, K_MAX = 3, 30
PCA_DIMS = 256
EXAMPLES = 12
SEED = 0


def _k_for(n_topics: int) -> int:

    return int(np.clip(n_topics // TOPICS_PER_CLUSTER, K_MIN, K_MAX))


def cluster_category(topics: pd.DataFrame, X: np.ndarray) -> pd.DataFrame:

    k = _k_for(len(topics))
    Z = PCA(
        n_components=min(PCA_DIMS, len(topics) - 1), random_state=SEED
    ).fit_transform(X)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    model = KMeans(n_clusters=k, n_init=4, random_state=SEED).fit(
        Z, sample_weight=topics["members"].to_numpy()
    )
    distance = np.linalg.norm(Z - model.cluster_centers_[model.labels_], axis=1)
    return topics.assign(cluster=model.labels_, distance=distance)


def main() -> None:

    topics = pd.read_parquet(DIR / "topics_categorised.parquet")
    X = np.load(DIR / "centroids.npy")
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    total = topics["members"].sum()

    parts, lines = [], ["# 002 — clusters inside each category (raw evidence)\n"]
    for slug in SLUGS:
        mask = (topics["category"] == slug).to_numpy()
        part = cluster_category(topics[mask], X[mask])
        part["cluster"] = slug + ":" + part["cluster"].astype(str)
        parts.append(part)

        category_articles = part["members"].sum()
        lines.append(
            f"\n## {slug} ({BY_SLUG[slug].name}) — {category_articles / total:.1%} of articles,"
            f" {len(part)} topics\n"
        )
        clusters = part.groupby("cluster").agg(
            articles=("members", "sum"),
            topics=("id", "size"),
            international=("scope", lambda s: (s == "internacional").mean()),
            confidence=("confidence", "mean"),
        )
        for name, row in clusters.sort_values("articles", ascending=False).iterrows():
            members = part[part["cluster"] == name]
            # the subjects nearest the centre, larger topics first among near ties
            examples = members.sort_values(
                ["distance", "members"], ascending=[True, False]
            )
            lines.append(
                f"### {name} — {row['articles'] / category_articles:.0%} of {slug},"
                f" {row['articles'] / total:.2%} of all; {row['topics']} topics;"
                f" international {row['international']:.0%}; confidence {row['confidence']:.2f}\n"
            )
            lines.extend(f"- {s}" for s in examples["subject"].head(EXAMPLES))
            lines.append("")

    out = pd.concat(parts).drop(columns="distance")
    out.to_parquet(DIR / "topics_clustered.parquet", index=False)
    (DIR / "clusters.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"wrote {out['cluster'].nunique()} clusters to {DIR / 'clusters.md'}"
        f" and {DIR / 'topics_clustered.parquet'}"
    )


if __name__ == "__main__":
    main()
