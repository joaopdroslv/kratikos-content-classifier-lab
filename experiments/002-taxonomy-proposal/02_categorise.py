"""Step 2 — put every topic under one of the 12 current categories with the 001 student.

The student is refit on all 3,075 labelled 001 articles (train + test) with the C that 001 chose,
then applied to each topic centroid. 001 measured it at 83% primary accuracy against a human, which
is enough to sort topics into buckets to cluster; the buckets are not an evaluation.

Also kept: the top probability, since a topic the student is unsure about is where a missing
master category would show up.

    uv run python experiments/002-taxonomy-proposal/02_categorise.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from lab.config import DATA_DIR
from lab.taxonomy import SLUGS

DIR = DATA_DIR / "002"
DIR_001 = DATA_DIR / "001"
K = len(SLUGS)
INDEX = {slug: i for i, slug in enumerate(SLUGS)}


def _normalise(X: np.ndarray) -> np.ndarray:

    return X / np.linalg.norm(X, axis=1, keepdims=True)


def train_student() -> OneVsRestClassifier:

    with (DIR_001 / "labels.jsonl").open(encoding="utf-8") as f:
        labels = pd.DataFrame([r for r in map(json.loads, f) if "error" not in r])
    labels = labels.drop_duplicates("id", keep="last")
    vectors = np.load(DIR_001 / "vectors.npz")
    by_id = dict(zip(vectors["ids"], vectors["vectors"]))
    labels = labels[labels["id"].isin(by_id)].reset_index(drop=True)

    X = _normalise(np.stack([by_id[id] for id in labels["id"]]))
    Y = np.zeros((len(labels), K), dtype=int)
    for row, (primary, secondary) in enumerate(
        zip(labels["primary"], labels["secondary"])
    ):
        for slug in ([primary] if isinstance(primary, str) else []) + list(secondary):
            Y[row, INDEX[slug]] = 1

    C = json.loads((DIR_001 / "results.json").read_text())["student"]["C"]
    print(f"student: {len(labels)} labelled articles, C={C}")
    return OneVsRestClassifier(LogisticRegression(C=C, max_iter=3_000), n_jobs=-1).fit(
        X, Y
    )


def main() -> None:

    topics = pd.read_parquet(DIR / "topics.parquet")
    X = _normalise(np.load(DIR / "centroids.npy"))
    scores = train_student().predict_proba(X)

    best = scores.argmax(axis=1)
    topics["category"] = [SLUGS[b] for b in best]
    topics["confidence"] = scores[np.arange(len(topics)), best]
    ranked = np.sort(scores, axis=1)
    topics["margin"] = ranked[:, -1] - ranked[:, -2]
    topics.to_parquet(DIR / "topics_categorised.parquet", index=False)
    np.save(DIR / "scores.npy", scores.astype(np.float32))

    weight = topics["members"]
    share = weight.groupby(topics["category"]).sum() / weight.sum()
    print("\narticle share per predicted category:")
    print(share.sort_values(ascending=False).round(3).to_string())
    low = topics["confidence"] < 0.5
    print(
        f"\nlow confidence (<0.5): {low.mean():.1%} of topics,"
        f" {weight[low].sum() / weight.sum():.1%} of articles"
    )


if __name__ == "__main__":
    main()
