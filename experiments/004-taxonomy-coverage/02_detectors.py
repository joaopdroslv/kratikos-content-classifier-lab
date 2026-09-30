"""Step 2 — can a distance to the prototypes tell a covered article from an uncovered one?

If it could, only the far part of the corpus would need an LLM. Tested on the 1,390 articles of
002's validation, where the teacher said whether a subcategory of the primary master fits
(`covered`) or not (no master, or `missing_subcategory` set). Scores, low = likely a gap:

- `max`: the highest cosine to any subcategory prototype;
- `z_max`: the same after standardising each prototype over the corpus (some definitions embed
  closer to everything than others);
- `margin`: best minus second-best subcategory cosine;
- `within_master`: best subcategory of the nearest master minus that master's own cosine (an
  article of a master that no subcategory fits sits nearer the master than any of its parts).

Also reported: how often the nearest prototypes agree with the teacher, i.e. whether they are
good enough to label clusters with in step 3.

The validation answers were given under taxonomy 2.0 and the prototypes are 2.1; the 2.1 edits
were merges and renames (`REMAP`, as in 002's step 8), plus the new `culture.religion`.

Output: `data/004/detectors.md`.

    uv run python experiments/004-taxonomy-coverage/02_detectors.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lab import llm
from lab.config import DATA_DIR

DIR = DATA_DIR / "004"
VALIDATION = (
    DATA_DIR / "002" / "validation_7cb234171d.jsonl"
)  # 002 step 7, taxonomy 2.0
CENTROIDS = DATA_DIR / "002" / "centroids.npy"
TOP_K = 3
# 2.0 subcategory -> 2.1, copied from 002's 08_review_sheet.py
REMAP = {
    "human_rights.racial_ethnic_religious": "human_rights.discrimination_equality",
    "human_rights.lgbtq": "human_rights.discrimination_equality",
    "human_rights.gender_women": "human_rights.discrimination_equality",
    "housing.rent": "housing.housing_access",
    "housing.social_housing": "housing.housing_access",
    "housing.homelessness": "housing.housing_access",
}


def normalise(X: np.ndarray) -> np.ndarray:

    return X / np.linalg.norm(X, axis=1, keepdims=True)


def load_prototypes(kind: str) -> tuple[list[str], np.ndarray]:

    prototypes = pd.read_parquet(DIR / "prototypes.parquet")
    part = prototypes[prototypes["kind"] == kind]
    return part["key"].tolist(), normalise(np.stack(part["vector"]))


def load_validation() -> tuple[pd.DataFrame, np.ndarray]:

    saved = np.load(DIR / "validation_vectors.npz")
    answers = pd.DataFrame(llm.read_jsonl(VALIDATION))
    df = pd.DataFrame({"id": saved["ids"], "row": range(len(saved["ids"]))}).merge(
        answers, on="id"
    )
    df["subcategories"] = df["subcategories"].map(
        lambda subs: {REMAP.get(s, s) for s in subs}
    )
    has_own_sub = df.apply(
        lambda r: any(s.startswith(f"{r['primary']}.") for s in r["subcategories"]),
        axis=1,
    )
    df["covered"] = (
        df["primary"].notna() & has_own_sub & df["missing_subcategory"].isna()
    )
    return df, normalise(saved["vectors"][df["row"]])


def scores(V: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:

    sub_keys, S = load_prototypes("subcategory")
    master_keys, M = load_prototypes("master")
    sims = V @ S.T
    corpus = normalise(np.load(CENTROIDS)) @ S.T
    ordered = np.sort(sims, axis=1)

    master_sims = V @ M.T
    nearest_master = master_sims.argmax(axis=1)
    sub_master = np.array([k.split(".")[0] for k in sub_keys])
    within = np.array(
        [
            sims[i, sub_master == master_keys[m]].max()
            for i, m in enumerate(nearest_master)
        ]
    )
    return (
        {
            "max": ordered[:, -1],
            "z_max": ((sims - corpus.mean(0)) / corpus.std(0)).max(axis=1),
            "margin": ordered[:, -1] - ordered[:, -2],
            "within_master": within - master_sims.max(axis=1),
        },
        np.argsort(-sims, axis=1)[:, :TOP_K],
        sub_keys,
    )


def main() -> None:

    df, V = load_validation()
    found, top, sub_keys = scores(V)
    uncovered = ~df["covered"].to_numpy()
    top_keys = [[sub_keys[i] for i in row] for row in top]

    lines = [
        "# 004 — distance-based gap detectors on 002's validation\n",
        f"{len(df)} articles: {(~uncovered).sum()} covered, {uncovered.sum()} uncovered.\n",
        "| score | AUC (0.5 = blind) | median covered | median uncovered |",
        "|---|---|---|---|",
    ]
    for name, s in found.items():
        lines.append(
            f"| `{name}` | {roc_auc_score(uncovered, -s):.2f}"
            f" | {np.median(s[~uncovered]):.3f} | {np.median(s[uncovered]):.3f} |"
        )
    top1_master = np.mean(
        [keys[0].split(".")[0] == p for keys, p in zip(top_keys, df["primary"])]
    )
    topk_sub = np.mean(
        [
            bool(set(keys) & subs)
            for keys, subs, c in zip(top_keys, df["subcategories"], df["covered"])
            if c
        ]
    )
    lines += [
        "",
        f"- Nearest subcategory prototype's master = the teacher's primary: {top1_master:.0%}.",
        f"- Covered articles with a teacher subcategory among the {TOP_K} nearest"
        f" prototypes: {topk_sub:.0%}.",
    ]
    report = "\n".join(lines)
    (DIR / "detectors.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
