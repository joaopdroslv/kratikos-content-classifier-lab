"""Step 4 — score the candidates against the teacher on a held-out, topic-grouped test split.

Candidates (all predict a `primary` and a label set over the 12 slugs):
- legacy: the category the news source wrote (`Geral` predicts nothing)
- zero_shot: cosine between the article vector and each category's definition vector
- student: one-vs-rest logistic regression on the article vectors, trained on teacher labels

Metrics:
- primary_acc: predicted primary == teacher primary, "no category" included as a class
- primary_in_set: predicted primary is one of the teacher's labels (items the teacher labelled)
- micro_f1 / macro_f1: the label sets, as multi-label
Reported on the whole test split and on its `uniform` stratum (the unbiased one).

    uv run python experiments/001-baseline-12-categories/04_evaluate.py
"""

import json
from collections import Counter

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier

from lab.config import DATA_DIR
from lab.taxonomy import CATEGORIES, SLUGS, slug_for_source_name

DIR = DATA_DIR / "001"
EMBEDDING_MODEL = "text-embedding-3-large"  # the model that produced the Qdrant vectors
C_GRID = (0.5, 2.0, 8.0, 32.0)
THRESHOLD_GRID = np.round(np.arange(0.05, 0.95, 0.05), 2)
LEARNING_CURVE_SIZES = (250, 500, 1_000)
SEED = 0

K = len(SLUGS)
INDEX = {slug: i for i, slug in enumerate(SLUGS)}


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
    X = np.stack([by_id[id] for id in df["id"]])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    print(f"{len(df)} labelled articles with vectors (sample {len(sample)})")
    return df, X


def label_matrix(df: pd.DataFrame) -> np.ndarray:

    Y = np.zeros((len(df), K), dtype=int)
    for row, (primary, secondary) in enumerate(zip(df["primary"], df["secondary"])):
        for slug in ([primary] if primary else []) + list(secondary):
            Y[row, INDEX[slug]] = 1
    return Y


# ---------------------------------------------------------------- metrics


def score(
    df: pd.DataFrame, Y: np.ndarray, pred_primary: list[str | None], pred_Y: np.ndarray
) -> dict:

    truth = df["primary"].tolist()
    labelled = [i for i, t in enumerate(truth) if t is not None]
    return {
        "n": len(df),
        "primary_acc": float(np.mean([p == t for p, t in zip(pred_primary, truth)])),
        "primary_in_set": float(
            np.mean(
                [
                    pred_primary[i] is not None and Y[i, INDEX[pred_primary[i]]]
                    for i in labelled
                ]
            )
        ),
        "micro_f1": float(f1_score(Y, pred_Y, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y, pred_Y, average="macro", zero_division=0)),
    }


def per_label(Y: np.ndarray, pred_Y: np.ndarray) -> pd.DataFrame:

    rows = []
    for slug, i in INDEX.items():
        tp = int((Y[:, i] & pred_Y[:, i]).sum())
        precision = tp / max(pred_Y[:, i].sum(), 1)
        recall = tp / max(Y[:, i].sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        rows.append((slug, int(Y[:, i].sum()), precision, recall, f1))
    return pd.DataFrame(rows, columns=["slug", "support", "precision", "recall", "f1"])


# ---------------------------------------------------------------- candidates


def predict_legacy(df: pd.DataFrame) -> tuple[list[str | None], np.ndarray]:

    primary = [slug_for_source_name(name) for name in df["source_category"]]
    Y = np.zeros((len(df), K), dtype=int)
    for row, slug in enumerate(primary):
        if slug:
            Y[row, INDEX[slug]] = 1
    return primary, Y


def label_vectors() -> np.ndarray:
    """One vector per category definition, cached — 12 embedding calls in total."""

    path = DIR / "label_vectors.npy"
    if path.exists():
        return np.load(path)
    texts = [f"{c.name}: {c.definition}" for c in CATEGORIES]
    response = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    V = np.array([d.embedding for d in response.data], dtype=np.float32)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    np.save(path, V)
    return V


def sets_from_scores(
    scores: np.ndarray, thresholds: np.ndarray, none_below: float
) -> tuple[list[str | None], np.ndarray]:
    """Primary = best-scoring label unless it is below `none_below`; the set = every label over its
    own threshold, plus the primary."""

    best = scores.argmax(axis=1)
    primary = [
        SLUGS[b] if scores[row, b] >= none_below else None for row, b in enumerate(best)
    ]
    Y = (scores >= thresholds).astype(int)
    for row, slug in enumerate(primary):
        if slug is None:
            Y[row] = 0
        else:
            Y[row, INDEX[slug]] = 1
    return primary, Y


def tune(scores: np.ndarray, Y: np.ndarray, truth: list[str | None], grid) -> tuple:
    """Per-label thresholds maximising that label's F1, then the no-category cut-off maximising
    primary accuracy — both on training (out-of-fold) scores, never on the test split.
    """

    thresholds = np.array(
        [
            max(
                grid,
                key=lambda t: f1_score(Y[:, i], scores[:, i] >= t, zero_division=0),
            )
            for i in range(K)
        ]
    )
    best = scores.argmax(axis=1)

    def accuracy(cut: float) -> float:
        return np.mean(
            [
                (SLUGS[b] if scores[r, b] >= cut else None) == t
                for r, (b, t) in enumerate(zip(best, truth))
            ]
        )

    none_below = max([0.0, *grid], key=accuracy)  # 0.0 = always predict a category
    return thresholds, float(none_below)


def zero_shot(
    X_train, Y_train, truth_train, X_test
) -> tuple[list[str | None], np.ndarray, dict]:

    V = label_vectors()
    train_scores, test_scores = X_train @ V.T, X_test @ V.T
    grid = np.round(np.arange(0.0, 0.6, 0.01), 2)
    thresholds, none_below = tune(train_scores, Y_train, truth_train, grid)
    primary, pred = sets_from_scores(test_scores, thresholds, none_below)
    return primary, pred, {"none_below": none_below}


def make_student(C: float) -> OneVsRestClassifier:

    return OneVsRestClassifier(LogisticRegression(C=C, max_iter=3_000), n_jobs=-1)


def student(
    X_train, Y_train, truth_train, groups_train, X_test
) -> tuple[list[str | None], np.ndarray, dict]:

    cv = GroupKFold(n_splits=4)

    def oof(C: float) -> np.ndarray:
        return cross_val_predict(
            make_student(C),
            X_train,
            Y_train,
            groups=groups_train,
            cv=cv,
            method="predict_proba",
        )

    candidates = {C: oof(C) for C in C_GRID}
    C, oof_scores = max(
        candidates.items(),
        key=lambda item: average_precision_score(Y_train, item[1], average="macro"),
    )
    thresholds, none_below = tune(oof_scores, Y_train, truth_train, THRESHOLD_GRID)
    model = make_student(C).fit(X_train, Y_train)
    primary, pred = sets_from_scores(
        model.predict_proba(X_test), thresholds, none_below
    )
    return (
        primary,
        pred,
        {
            "C": C,
            "none_below": none_below,
            "thresholds": dict(zip(SLUGS, thresholds.tolist())),
            "model": model,
        },
    )


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
    Y = label_matrix(df)
    teacher_report(df)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, groups=df["split_group"]))
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
    student_primary, student_Y, student_info = student(
        X_train, Y_train, truth_train, train["split_group"].to_numpy(), X_test
    )
    predictions["student"] = (student_primary, student_Y)

    results = {}
    uniform_mask = (test["stratum"] == "uniform").to_numpy()
    print("\n## Candidates vs teacher (test split)")
    for name, (primary, pred) in predictions.items():
        whole = score(test, Y_test, primary, pred)
        uni = score(
            test[uniform_mask],
            Y_test[uniform_mask],
            [p for p, m in zip(primary, uniform_mask) if m],
            pred[uniform_mask],
        )
        results[name] = {"test": whole, "test_uniform": uni}
        print(
            f"  {name:10s} primary_acc {whole['primary_acc']:.3f} (uniform {uni['primary_acc']:.3f})"
            f"  primary_in_set {whole['primary_in_set']:.3f}"
            f"  micro_f1 {whole['micro_f1']:.3f}  macro_f1 {whole['macro_f1']:.3f}"
        )

    print(f"\nstudent: C={student_info['C']}  none_below={student_info['none_below']}")
    report = per_label(Y_test, student_Y)
    print(report.round(3).to_string(index=False))

    print("\n## Learning curve (student, fixed C, default 0.5 thresholds)")
    rng = np.random.default_rng(SEED)
    curve = {}
    for size in (*LEARNING_CURVE_SIZES, len(train)):
        idx = rng.choice(len(train), size=min(size, len(train)), replace=False)
        model = make_student(student_info["C"]).fit(X_train[idx], Y_train[idx])
        primary, pred = sets_from_scores(
            model.predict_proba(X_test), np.full(K, 0.5), student_info["none_below"]
        )
        curve[size] = score(test, Y_test, primary, pred)
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
