"""Resumable, concurrent structured-output calls — the loop of 001's `03_label.py`, made generic.

Every answer is appended to a JSONL file as it arrives, keyed by `id`; ids already there without an
error are skipped, so an interrupted run is simply run again.
"""

import asyncio
import json
from collections.abc import Iterable
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

ATTEMPTS = 4


def done_ids(path: Path) -> set[str]:

    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {r["id"] for r in map(json.loads, f) if "error" not in r}


def read_jsonl(path: Path) -> list[dict]:
    """The last successful record per id."""

    with path.open(encoding="utf-8") as f:
        records = {r["id"]: r for r in map(json.loads, f) if "error" not in r}
    return list(records.values())


async def _one(
    client: AsyncOpenAI,
    model: str,
    system: str,
    item_id: str,
    user: str,
    schema: type[BaseModel],
    semaphore: asyncio.Semaphore,
    sink,
    extra: dict,
) -> None:

    async with semaphore:
        for attempt in range(ATTEMPTS):
            try:
                completion = await client.chat.completions.parse(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format=schema,
                )
                record = {
                    "id": item_id,
                    **completion.choices[0].message.parsed.model_dump(),
                    "model": model,
                    **extra,
                }
                break
            except Exception as error:
                if attempt == ATTEMPTS - 1:
                    record = {
                        "id": item_id,
                        "error": f"{type(error).__name__}: {error}"[:500],
                    }
                    break
                await asyncio.sleep(2**attempt)
        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        sink.flush()


async def run(
    items: Iterable[tuple[str, str]],
    *,
    out: Path,
    system: str,
    schema: type[BaseModel],
    model: str,
    concurrency: int = 16,
    extra: dict | None = None,
    desc: str = "llm",
) -> None:
    """Calls `model` once per `(id, user_message)` not already in `out`."""

    done = done_ids(out)
    todo = [(item_id, user) for item_id, user in items if item_id not in done]
    print(f"{desc}: model={model}  done={len(done)}  to do={len(todo)}")
    if not todo:
        return
    semaphore = asyncio.Semaphore(concurrency)
    out.parent.mkdir(parents=True, exist_ok=True)
    # closed inside the loop: an unclosed client is torn down after asyncio.run() returns
    async with AsyncOpenAI() as client:
        with out.open("a", encoding="utf-8") as sink:
            await tqdm.gather(
                *(
                    _one(
                        client,
                        model,
                        system,
                        i,
                        u,
                        schema,
                        semaphore,
                        sink,
                        extra or {},
                    )
                    for i, u in todo
                ),
                desc=desc,
            )


def render_article(row) -> str:
    """Title + description + the start of the content, as 001 fed the teacher."""

    parts = [f"Title: {row['title']}"]
    if row.get("description"):
        parts.append(f"Description: {row['description']}")
    if row.get("content"):
        parts.append(f"Content: {row['content']}")
    return "\n\n".join(parts)
