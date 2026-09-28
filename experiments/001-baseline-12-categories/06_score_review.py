"""Step 6 — score the teacher, the student and the legacy category against the human review.

The truth of a reviewed row is `correct_primary` when it is filled (`none` meaning no category
fits), otherwise the teacher's primary, which the reviewer accepted (`teacher_ok` = y/1).

The sheet is joined back to `test_predictions.parquet` by id and only the reviewer's three columns
are read from it: a spreadsheet round-trip can lose or mangle the other columns (Excel in pt-BR
re-saves with `;`, and 14 rows of the 001 sheet came back with the teacher's columns blank). A row
the reviewer judged without seeing the teacher is still valid, and is counted apart as `blind`.

    uv run python experiments/001-baseline-12-categories/06_score_review.py
"""

import math

import pandas as pd

from lab.config import DATA_DIR

DIR = DATA_DIR / "001"
YES = {"1", "1.0", "y", "yes", "s", "sim"}
NO = {"0", "0.0", "n", "no", "nao", "não"}
NONE = "none"


def _wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:

    if n == 0:
        return (math.nan, math.nan)
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def load() -> pd.DataFrame:

    sheet = pd.read_csv(
        DIR / "review.csv", sep=None, engine="python", encoding="utf-8-sig", dtype=str
    )[["id", "teacher_ok", "correct_primary", "notes"]]
    predictions = pd.read_parquet(DIR / "test_predictions.parquet")
    df = sheet.merge(predictions, on="id", how="left", validate="one_to_one")
    if df["primary"].isna().all():
        raise RuntimeError("no reviewed id matched test_predictions.parquet")

    ok = df["teacher_ok"].fillna("").str.strip().str.lower()
    correct = df["correct_primary"].fillna("").str.strip().str.lower()
    df["blind"] = ~ok.isin(YES | NO)
    unjudged = df["blind"] & (correct == "")
    if unjudged.any():
        print(
            f"skipping {int(unjudged.sum())} rows with neither teacher_ok nor correct_primary"
        )
        df, correct = df[~unjudged], correct[~unjudged]

    # "No category" is the string NONE, not None: pandas compares None != None element-wise.
    teacher = df["primary"].fillna(NONE)
    df["human_primary"] = correct.where(correct != "", teacher)
    return df.reset_index(drop=True)


def main() -> None:

    df = load()
    print(f"{len(df)} reviewed articles ({int(df['blind'].sum())} judged blind)\n")

    rows = []
    for name, column in (
        ("teacher", "primary"),
        ("student", "student_primary"),
        ("legacy", "legacy_primary"),
    ):
        hits = df[column].fillna(NONE) == df["human_primary"]
        low, high = _wilson(int(hits.sum()), len(df))
        blind = hits[df["blind"]]
        rows.append(
            (name, int(hits.sum()), len(df), hits.mean(), low, high, blind.mean())
        )
    table = pd.DataFrame(
        rows,
        columns=["candidate", "hits", "n", "primary_acc", "ci_low", "ci_high", "blind"],
    )
    print("Primary vs human (95% Wilson interval):")
    print(table.round(3).to_string(index=False))

    wrong = df[df["primary"].fillna(NONE) != df["human_primary"]]
    print(f"\nTeacher misses ({len(wrong)}):")
    print(
        wrong[["title", "primary", "human_primary", "student_primary"]]
        .assign(title=wrong["title"].str.slice(0, 70))
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
