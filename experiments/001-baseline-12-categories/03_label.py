"""Step 3 — label the sample with the teacher LLM.

Structured output over a closed enum of the 12 slugs: one `primary` category, up to two
`secondary` ones, and — when none of the 12 fits — a null primary plus a free-text
`missing_category`. That last field is the taxonomy signal of this experiment: it counts how much of
the corpus the current categories cannot hold, and names what is missing.

Resumable: every answer is appended to labels.jsonl as it arrives, and ids already there are
skipped, so an interrupted run is simply run again.

    LABEL_MODEL=gpt-4.1-mini uv run python experiments/001-baseline-12-categories/03_label.py [--limit N]
"""

import argparse
import asyncio
import json
import os
from typing import Literal

import pandas as pd
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from lab.config import DATA_DIR
from lab.taxonomy import CATEGORIES, SLUGS

SAMPLE = DATA_DIR / "001" / "sample.parquet"
OUT = DATA_DIR / "001" / "labels.jsonl"
MODEL = os.getenv("LABEL_MODEL", "gpt-4.1-mini")
CONCURRENCY = 16
PROMPT_VERSION = "001-v1"

Slug = Literal[SLUGS]


class Labeling(BaseModel):

    primary: Slug | None
    secondary: list[Slug]
    missing_category: str | None


_DEFINITIONS = "\n".join(f"- {c.slug} ({c.name}): {c.definition}" for c in CATEGORIES)

SYSTEM_PROMPT = f"""You classify news articles for a Brazilian civic-debate app.

Categories:
{_DEFINITIONS}

Rules:
- `primary`: the single category the article is mainly about.
- `secondary`: other categories the article substantially covers (0 to 2). A passing mention is not
  enough. Never repeat the primary.
- If none of the categories fits the article's main subject, set `primary` to null, leave
  `secondary` empty, and set `missing_category` to a short Brazilian Portuguese name for the
  category that would fit. Otherwise `missing_category` is null.
- Judge by the content, not by the outlet or by where the article happens to have been filed.
- Articles may be in any language."""


def _render(row: pd.Series) -> str:

    parts = [f"Title: {row['title']}"]
    if row["description"]:
        parts.append(f"Description: {row['description']}")
    if row["content"]:
        parts.append(f"Content: {row['content']}")
    return "\n\n".join(parts)


def _normalise(labeling: Labeling) -> dict:

    if labeling.primary is None:
        return {
            "primary": None,
            "secondary": [],
            "missing_category": labeling.missing_category,
        }
    secondary = [s for s in dict.fromkeys(labeling.secondary) if s != labeling.primary][
        :2
    ]
    return {
        "primary": labeling.primary,
        "secondary": secondary,
        "missing_category": None,
    }


async def _label_one(
    client: AsyncOpenAI, row: pd.Series, semaphore: asyncio.Semaphore, sink
) -> None:

    async with semaphore:
        for attempt in range(4):
            try:
                completion = await client.chat.completions.parse(
                    model=MODEL,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _render(row)},
                    ],
                    response_format=Labeling,
                )
                labeling = completion.choices[0].message.parsed
                record = {
                    "id": row["id"],
                    **_normalise(labeling),
                    "model": MODEL,
                    "prompt_version": PROMPT_VERSION,
                }
                break
            except Exception as error:
                if attempt == 3:
                    record = {
                        "id": row["id"],
                        "error": f"{type(error).__name__}: {error}"[:500],
                    }
                    break
                await asyncio.sleep(2**attempt)
        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        sink.flush()


async def main(limit: int | None) -> None:

    sample = pd.read_parquet(SAMPLE)
    done: set[str] = set()
    if OUT.exists():
        with OUT.open(encoding="utf-8") as f:
            done = {r["id"] for r in map(json.loads, f) if "error" not in r}
    todo = sample[~sample["id"].isin(done)]
    if limit:
        todo = todo.head(limit)
    print(f"model={MODEL}  already labelled={len(done)}  to label={len(todo)}")

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    with OUT.open("a", encoding="utf-8") as sink:
        await tqdm.gather(
            *(_label_one(client, row, semaphore, sink) for _, row in todo.iterrows()),
            desc="labelling",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    asyncio.run(main(parser.parse_args().limit))
