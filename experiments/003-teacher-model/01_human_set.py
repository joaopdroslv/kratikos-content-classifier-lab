"""Step 1 — candidate teachers on the 150 articles of the 001 human review.

Each candidate labels the reviewed articles with 001's exact prompt and schema (`03_label.py`,
prompt_version 001-v1, the 12 categories), and its primary is scored against the human primary
from `06_score_review.py`. The reference is `gpt-4.1-mini`, 001's teacher, whose answers are
already in `data/001/labels.jsonl`.

Caveat: the reviewer saw gpt-4.1-mini's answer and accepted it on most rows, so the truth leans
toward it on ambiguous articles; the misses of each candidate are printed to be read, not only
counted.

    uv run python experiments/003-teacher-model/01_human_set.py
"""

import asyncio
import importlib
import json
import sys
from pathlib import Path

import pandas as pd

from lab import llm
from lab.config import DATA_DIR
from lab.pricing import cost

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "001-baseline-12-categories"))
label_001 = importlib.import_module("03_label")
review_001 = importlib.import_module("06_score_review")

DIR = DATA_DIR / "003"
NONE = review_001.NONE
# name: (model, reasoning_effort)
CANDIDATES = {
    "gpt-6-luna_none": ("gpt-6-luna", "none"),
    "gpt-6-luna_low": ("gpt-6-luna", "low"),
}


async def label(reviewed: pd.DataFrame) -> None:

    sample = pd.read_parquet(DATA_DIR / "001" / "sample.parquet")
    rows = sample[sample["id"].isin(reviewed["id"])].to_dict("records")
    for name, (model, effort) in CANDIDATES.items():
        await llm.run(
            [(r["id"], llm.render_article(r)) for r in rows],
            out=DIR / f"human_{name}.jsonl",
            system=label_001.SYSTEM_PROMPT,
            schema=label_001.Labeling,
            model=model,
            reasoning_effort=effort,
            extra={
                "prompt_version": label_001.PROMPT_VERSION,
                "taxonomy_version": label_001.TAXONOMY_VERSION,
            },
            desc=name,
        )


def main() -> None:

    reviewed = review_001.load()
    asyncio.run(label(reviewed))

    truth = reviewed.set_index("id")["human_primary"]
    answers = {"gpt-4.1-mini (001)": reviewed.set_index("id")["primary"]}
    usage = {}
    for name, (model, _) in CANDIDATES.items():
        records = llm.read_jsonl(DIR / f"human_{name}.jsonl")
        answers[name] = pd.Series({r["id"]: r["primary"] for r in records})
        usage[name] = (model, [r["usage"] for r in records])

    titles = reviewed.set_index("id")["title"]
    rows, misses = [], {}
    for name, primary in answers.items():
        # a failed call has no answer
        ids = truth.index[truth.index.isin(primary.index)]
        answer = primary.reindex(ids).fillna(NONE)
        hits = answer == truth[ids]
        low, high = review_001._wilson(int(hits.sum()), len(ids))
        spent = cost(*usage[name]) if name in usage else float("nan")
        rows.append((name, int(hits.sum()), len(ids), hits.mean(), low, high, spent))
        wrong = ids[~hits.to_numpy()]
        misses[name] = pd.DataFrame(
            {
                "title": titles[wrong].str[:70],
                "human": truth[wrong],
                "answer": answer[wrong],
            }
        )

    table = pd.DataFrame(
        rows,
        columns=["candidate", "hits", "n", "primary_acc", "ci_low", "ci_high", "usd"],
    )
    print("\nPrimary vs human, 001 review (95% Wilson interval):")
    print(table.round(4).to_string(index=False))
    for name, df in misses.items():
        print(f"\n{name} misses ({len(df)}):")
        print(df.to_string(index=False))

    with (DIR / "human_results.json").open("w", encoding="utf-8") as f:
        json.dump(table.to_dict("records"), f, indent=2)


if __name__ == "__main__":
    main()
