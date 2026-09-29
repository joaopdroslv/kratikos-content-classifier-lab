"""Metrics for a candidate that predicts a primary label and a label set, against a reference.

- primary_acc: predicted primary == reference primary, "no category" (None) included as a class
- primary_in_set: predicted primary is one of the reference's labels (items the reference labelled)
- micro_f1 / macro_f1: the label sets, as multi-label
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def score(
    truth: Sequence[str | None],
    Y: np.ndarray,
    pred_primary: Sequence[str | None],
    pred_Y: np.ndarray,
    slugs: Sequence[str],
) -> dict:

    index = {slug: i for i, slug in enumerate(slugs)}
    labelled = [i for i, t in enumerate(truth) if t is not None]
    return {
        "n": len(truth),
        "primary_acc": float(np.mean([p == t for p, t in zip(pred_primary, truth)])),
        "primary_in_set": float(
            np.mean(
                [
                    pred_primary[i] is not None and Y[i, index[pred_primary[i]]]
                    for i in labelled
                ]
            )
        ),
        "micro_f1": float(f1_score(Y, pred_Y, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(Y, pred_Y, average="macro", zero_division=0)),
    }


def per_label(Y: np.ndarray, pred_Y: np.ndarray, slugs: Sequence[str]) -> pd.DataFrame:

    rows = []
    for i, slug in enumerate(slugs):
        tp = int((Y[:, i] & pred_Y[:, i]).sum())
        precision = tp / max(pred_Y[:, i].sum(), 1)
        recall = tp / max(Y[:, i].sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        rows.append((slug, int(Y[:, i].sum()), precision, recall, f1))
    return pd.DataFrame(rows, columns=["slug", "support", "precision", "recall", "f1"])
