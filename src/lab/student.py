"""The student and what it shares with the other candidates: label matrices, thresholds, training.

Taxonomy-agnostic: every function takes the ordered label `slugs` (the 001 masters, the v2
masters, or `master.subcategory` keys), and column `i` of every matrix is `slugs[i]`.

A label set is derived from per-label scores (a probability, or a cosine) the same way for every
candidate: the primary is the best-scoring label unless it is below `none_below` ("no category"),
and the set is every label over its own threshold, plus the primary. Thresholds and `none_below`
are tuned on training scores (out-of-fold for the student), never on the test split.
"""

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier

C_GRID = (0.5, 2.0, 8.0, 32.0)
THRESHOLD_GRID = np.round(np.arange(0.05, 0.95, 0.05), 2)


def normalise(X: np.ndarray) -> np.ndarray:
    """Unit-length rows, so a dot product is a cosine."""

    return X / np.linalg.norm(X, axis=1, keepdims=True)


def label_matrix(
    primaries: Sequence[str | None],
    secondaries: Sequence[Sequence[str]],
    slugs: Sequence[str],
) -> np.ndarray:
    """One row per item, 1 for its primary and each secondary label."""

    index = {slug: i for i, slug in enumerate(slugs)}
    Y = np.zeros((len(primaries), len(slugs)), dtype=int)
    for row, (primary, secondary) in enumerate(zip(primaries, secondaries)):
        for slug in ([primary] if isinstance(primary, str) else []) + list(secondary):
            Y[row, index[slug]] = 1
    return Y


def topic_split(
    groups: Sequence, test_size: float = 0.25, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Train and test row indices, with every topic (`split_group`) wholly on one side, so the test
    split never holds near-duplicates of training articles."""

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(splitter.split(np.zeros(len(groups)), groups=groups))


def sets_from_scores(
    scores: np.ndarray,
    thresholds: np.ndarray,
    none_below: float,
    slugs: Sequence[str],
) -> tuple[list[str | None], np.ndarray]:
    """Primary = best-scoring label unless it is below `none_below`; the set = every label over its
    own threshold, plus the primary."""

    best = scores.argmax(axis=1)
    primary = [
        slugs[b] if scores[row, b] >= none_below else None for row, b in enumerate(best)
    ]
    Y = (scores >= thresholds).astype(int)
    for row, b in enumerate(best):
        if primary[row] is None:
            Y[row] = 0
        else:
            Y[row, b] = 1
    return primary, Y


def tune(
    scores: np.ndarray,
    Y: np.ndarray,
    truth: Sequence[str | None],
    grid: Sequence[float],
    slugs: Sequence[str],
) -> tuple[np.ndarray, float]:
    """Per-label thresholds maximising that label's F1, then the no-category cut-off maximising
    primary accuracy — both on training (out-of-fold) scores, never on the test split.
    """

    thresholds = np.array(
        [
            max(
                grid,
                key=lambda t: f1_score(Y[:, i], scores[:, i] >= t, zero_division=0),
            )
            for i in range(len(slugs))
        ]
    )
    best = scores.argmax(axis=1)

    def accuracy(cut: float) -> float:
        return np.mean(
            [
                (slugs[b] if scores[r, b] >= cut else None) == t
                for r, (b, t) in enumerate(zip(best, truth))
            ]
        )

    none_below = max([0.0, *grid], key=accuracy)  # 0.0 = always predict a category
    return thresholds, float(none_below)


def make_student(C: float) -> OneVsRestClassifier:
    """One-vs-rest logistic regression, one binary classifier per label."""

    return OneVsRestClassifier(LogisticRegression(C=C, max_iter=3_000), n_jobs=-1)


def select_student(
    X: np.ndarray,
    Y: np.ndarray,
    truth: Sequence[str | None],
    groups: Sequence,
    slugs: Sequence[str],
    C_grid: Sequence[float] = C_GRID,
    threshold_grid: Sequence[float] = THRESHOLD_GRID,
) -> dict:
    """Pick `C` by macro average precision on topic-grouped out-of-fold scores, tune the thresholds
    on those same scores, then fit on all of `X`.

    Returns `C`, `none_below`, `thresholds` (per slug) and the fitted `model`.
    """

    cv = GroupKFold(n_splits=4)

    def oof(C: float) -> np.ndarray:
        return cross_val_predict(
            make_student(C), X, Y, groups=groups, cv=cv, method="predict_proba"
        )

    candidates = {C: oof(C) for C in C_grid}
    C, oof_scores = max(
        candidates.items(),
        key=lambda item: average_precision_score(Y, item[1], average="macro"),
    )
    thresholds, none_below = tune(oof_scores, Y, truth, threshold_grid, slugs)
    return {
        "C": C,
        "none_below": none_below,
        "thresholds": dict(zip(slugs, thresholds.tolist())),
        "model": make_student(C).fit(X, Y),
    }
