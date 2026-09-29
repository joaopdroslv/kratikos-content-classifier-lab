"""Step 4 — score the candidates against the teacher on a held-out, topic-grouped test split.

Candidates (all predict a `primary` and a label set over the 12 slugs):
- legacy: the category the news source wrote (`Geral` predicts nothing)
- zero_shot: cosine between the article vector and each category's definition vector
- student: one-vs-rest logistic regression on the article vectors, trained on teacher labels

Metrics (`lab.metrics`, with the teacher as reference): primary_acc, primary_in_set, micro_f1,
macro_f1. Reported on the whole test split and on its `uniform` stratum (the unbiased one).

    uv run python experiments/001-baseline-12-categories/04_evaluate.py
"""

import json
from collections import Counter

import numpy as np
import pandas as pd
from openai import OpenAI

from lab.config import DATA_DIR
from lab.metrics import per_label, score
from lab.student import (
    label_matrix,
    make_student,
    normalise,
    select_student,
    sets_from_scores,
    topic_split,
    tune,
)
from lab.taxonomy import CATEGORIES, SLUGS, slug_for_source_name

DIR = DATA_DIR / "001"
EMBEDDING_MODEL = "text-embedding-3-large"  # the model that produced the Qdrant vectors
LEARNING_CURVE_SIZES = (250, 500, 1_000)
SEED = 0

K = len(SLUGS)


# ---------------------------------------------------------------- data


def load() -> tuple[pd.DataFrame, np.ndarray]:

    sample = pd.read_parquet(DIR / "sample.parquet")
    with (DIR / "labels.jsonl").open(encoding="utf-8") as f:
        labels = pd.DataFrame([r for r in map(json.loads, f) if "error" not in r])
    labels = labels.drop_duplicates("id", keep="last")

    vectors = np.load(DIR / "vectors.npz")
    by_id = dict(zip(vectors["ids"], vectors["vectors"]))

    df = sample.merge(
        labels[["id", "primary", "secondary", "missing_category"]], on="id"
    )
    df = df[df["id"].isin(by_id)].reset_index(drop=True)
    # pandas reads a JSON null as NaN; "no category" must stay None to compare equal to None.
    df["primary"] = df["primary"].astype(object).where(df["primary"].notna(), None)
    X = normalise(np.stack([by_id[id] for id in df["id"]]))
    print(f"{len(df)} labelled articles with vectors (sample {len(sample)})")
    return df, X


# ---------------------------------------------------------------- candidates


def predict_legacy(df: pd.DataFrame) -> tuple[list[str | None], np.ndarray]:

    primary = [slug_for_source_name(name) for name in df["source_category"]]
    return primary, label_matrix(primary, [[]] * len(primary), SLUGS)


def label_vectors() -> np.ndarray:
    """One vector per category definition, cached — 12 embedding calls in total."""

    path = DIR / "label_vectors.npy"
    if path.exists():
        return np.load(path)
    texts = [f"{c.name}: {c.definition}" for c in CATEGORIES]
    response = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    V = normalise(np.array([d.embedding for d in response.data], dtype=np.float32))
    np.save(path, V)
    return V


def zero_shot(
    X_train, Y_train, truth_train, X_test
) -> tuple[list[str | None], np.ndarray, dict]:

    V = label_vectors()
    train_scores, test_scores = X_train @ V.T, X_test @ V.T
    grid = np.round(np.arange(0.0, 0.6, 0.01), 2)
    thresholds, none_below = tune(train_scores, Y_train, truth_train, grid, SLUGS)
    primary, pred = sets_from_scores(test_scores, thresholds, none_below, SLUGS)
    return primary, pred, {"none_below": none_below}


# ---------------------------------------------------------------- report


def teacher_report(df: pd.DataFrame) -> None:

    uniform = df[df["stratum"] == "uniform"]
    print("\n## Teacher, uniform stratum")
    print(f"no category fits: {uniform['primary'].isna().mean():.1%}")
    print(
        uniform["primary"]
        .fillna("(none)")
        .value_counts(normalize=True)
        .round(3)
        .to_string()
    )
    print("secondary labels per article:")
    print(
        uniform["secondary"].map(len).value_counts(normalize=True).round(3).to_string()
    )
    missing = Counter(
        m.strip().lower() for m in df["missing_category"].dropna() if m.strip()
    )
    print("\nmissing_category suggestions (whole sample):")
    for name, count in missing.most_common(25):
        print(f"  {count:4d}  {name}")

    print(
        "\n## Legacy vs teacher, uniform stratum: where each source category really goes"
    )
    table = pd.crosstab(
        uniform["source_category"],
        uniform["primary"].fillna("(none)"),
        normalize="index",
    )
    agreement = {
        name: table.loc[name].get(slug_for_source_name(name) or "(none)", 0.0)
        for name in table.index
    }
    counts = uniform["source_category"].value_counts()
    for name, share in sorted(agreement.items(), key=lambda kv: -counts[kv[0]]):
        top = table.loc[name].sort_values(ascending=False).head(3)
        spread = ", ".join(f"{slug} {v:.0%}" for slug, v in top.items())
        print(f"  {name:18s} n={counts[name]:4d}  agrees {share:5.1%}  -> {spread}")


def main() -> None:

    df, X = load()
    Y = label_matrix(df["primary"], df["secondary"], SLUGS)
    teacher_report(df)

    train_idx, test_idx = topic_split(df["split_group"], seed=SEED)
    train, test = df.iloc[train_idx], df.iloc[test_idx].reset_index(drop=True)
    X_train, X_test, Y_train, Y_test = (
        X[train_idx],
        X[test_idx],
        Y[train_idx],
        Y[test_idx],
    )
    truth_train = train["primary"].tolist()
    print(f"\ntrain={len(train)}  test={len(test)}  (grouped by topic)")

    predictions = {
        "legacy": predict_legacy(test),
        "zero_shot": zero_shot(X_train, Y_train, truth_train, X_test)[:2],
    }
    student_info = select_student(
        X_train, Y_train, truth_train, train["split_group"].to_numpy(), SLUGS
    )
    student_primary, student_Y = sets_from_scores(
        student_info["model"].predict_proba(X_test),
        np.array(list(student_info["thresholds"].values())),
        student_info["none_below"],
        SLUGS,
    )
    predictions["student"] = (student_primary, student_Y)

    results = {}
    uniform_mask = (test["stratum"] == "uniform").to_numpy()
    truth_test = test["primary"].tolist()
    print("\n## Candidates vs teacher (test split)")
    for name, (primary, pred) in predictions.items():
        whole = score(truth_test, Y_test, primary, pred, SLUGS)
        uni = score(
            [t for t, m in zip(truth_test, uniform_mask) if m],
            Y_test[uniform_mask],
            [p for p, m in zip(primary, uniform_mask) if m],
            pred[uniform_mask],
            SLUGS,
        )
        results[name] = {"test": whole, "test_uniform": uni}
        print(
            f"  {name:10s} primary_acc {whole['primary_acc']:.3f} (uniform {uni['primary_acc']:.3f})"
            f"  primary_in_set {whole['primary_in_set']:.3f}"
            f"  micro_f1 {whole['micro_f1']:.3f}  macro_f1 {whole['macro_f1']:.3f}"
        )

    print(f"\nstudent: C={student_info['C']}  none_below={student_info['none_below']}")
    report = per_label(Y_test, student_Y, SLUGS)
    print(report.round(3).to_string(index=False))

    print("\n## Learning curve (student, fixed C, default 0.5 thresholds)")
    rng = np.random.default_rng(SEED)
    curve = {}
    for size in (*LEARNING_CURVE_SIZES, len(train)):
        idx = rng.choice(len(train), size=min(size, len(train)), replace=False)
        model = make_student(student_info["C"]).fit(X_train[idx], Y_train[idx])
        primary, pred = sets_from_scores(
            model.predict_proba(X_test),
            np.full(K, 0.5),
            student_info["none_below"],
            SLUGS,
        )
        curve[size] = score(truth_test, Y_test, primary, pred, SLUGS)
        print(
            f"  n={size:5d}  primary_acc {curve[size]['primary_acc']:.3f}"
            f"  micro_f1 {curve[size]['micro_f1']:.3f}  macro_f1 {curve[size]['macro_f1']:.3f}"
        )

    out = test[
        [
            "id",
            "stratum",
            "title",
            "description",
            "source_category",
            "primary",
            "secondary",
        ]
    ]
    out = out.assign(
        student_primary=student_primary,
        student_labels=[[SLUGS[i] for i in np.flatnonzero(row)] for row in student_Y],
        legacy_primary=predictions["legacy"][0],
    )
    out.to_parquet(DIR / "test_predictions.parquet", index=False)
    del student_info["model"]
    with (DIR / "results.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"results": results, "student": student_info, "learning_curve": curve},
            f,
            indent=2,
        )
    print(f"\nwrote {DIR / 'results.json'} and {DIR / 'test_predictions.parquet'}")


if __name__ == "__main__":
    main()
