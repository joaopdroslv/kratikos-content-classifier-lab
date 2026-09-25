"""Step 5 — a sheet for a human to judge the teacher (and the student) on uniform test articles.

The teacher is the ceiling of everything else here, so its own error rate has to be measured by a
person. Fill `teacher_ok` (y/n) and, when it is n, `correct_primary` with a slug or `none`. The
filled sheet becomes the first golden set, and is never used for training.

    uv run python experiments/001-baseline-12-categories/05_review_sheet.py
"""

import pandas as pd

from lab.config import DATA_DIR

DIR = DATA_DIR / "001"
SIZE = 150


def main() -> None:

    predictions = pd.read_parquet(DIR / "test_predictions.parquet")
    uniform = predictions[predictions["stratum"] == "uniform"]
    sheet = uniform.sample(n=min(SIZE, len(uniform)), random_state=0)
    sheet = sheet.assign(
        secondary=sheet["secondary"].map(", ".join),
        student_labels=sheet["student_labels"].map(", ".join),
        teacher_ok="",
        correct_primary="",
        notes="",
    )[
        [
            "id",
            "title",
            "description",
            "source_category",
            "primary",
            "secondary",
            "student_primary",
            "student_labels",
            "teacher_ok",
            "correct_primary",
            "notes",
        ]
    ]
    path = DIR / "review.csv"
    # BOM so Excel reads the accents
    sheet.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"wrote {len(sheet)} rows to {path}")


if __name__ == "__main__":
    main()
