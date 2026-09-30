"""Resumable, concurrent structured-output calls — the loop of 001's `03_label.py`, made generic.

Every answer is appended to a JSONL file as it arrives, keyed by `id`; ids already there without an
error are skipped, so an interrupted run is simply run again. Every answer carries `model`,
`created_at` and the token `usage` (to compare cost); the caller adds `prompt_version` (and any
other metadata) through `extra`.

Temperature is 0, except for reasoning models at a `reasoning_effort` above "none", which only
accept the default: those answers are not deterministic, and carry `reasoning_effort`.
"""

import asyncio
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.asyncio import tqdm

from lab import metadata

ATTEMPTS = 4


def done_ids(path: Path) -> set[str]:

    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {r["id"] for r in map(json.loads, f) if "error" not in r}


def output_path(directory: Path, prefix: str, system: str) -> Path:
    """`<prefix>_<hash of the system prompt>.jsonl`: answers to another prompt (a changed
    taxonomy, a reworded rule) never mix into the same file."""

    return (
        directory / f"{prefix}_{hashlib.sha256(system.encode()).hexdigest()[:10]}.jsonl"
    )


def read_jsonl(path: Path) -> list[dict]:
    """The last successful record per id."""

    with path.open(encoding="utf-8") as f:
        records = {r["id"]: r for r in map(json.loads, f) if "error" not in r}
    return list(records.values())


def _usage(usage) -> dict:

    prompt = usage.prompt_tokens_details
    completion = usage.completion_tokens_details
    return {
        "input": usage.prompt_tokens,
        "cached": (prompt.cached_tokens or 0) if prompt else 0,
        "output": usage.completion_tokens,
        "reasoning": (completion.reasoning_tokens or 0) if completion else 0,
    }


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
    reasoning_effort: str | None,
) -> None:

    sampling = {"temperature": 0}
    if reasoning_effort is not None:
        sampling = {"reasoning_effort": reasoning_effort}
        if reasoning_effort == "none":
            sampling["temperature"] = 0
        extra = {**extra, "reasoning_effort": reasoning_effort}
    async with semaphore:
        for attempt in range(ATTEMPTS):
            try:
                completion = await client.chat.completions.parse(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    response_format=schema,
                    **sampling,
                )
                record = {
                    "id": item_id,
                    **completion.choices[0].message.parsed.model_dump(),
                    "model": model,
                    **extra,
                    "usage": _usage(completion.usage),
                    "created_at": metadata.now(),
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
    reasoning_effort: str | None = None,
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
                        reasoning_effort,
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
