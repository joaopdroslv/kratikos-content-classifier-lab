"""The teacher's classification on taxonomy v2: prompt, answer schema and answer format.

The one format every experiment uses to label articles with `lab.taxonomy_v2`, so labels from
different experiments (validation, teacher comparison, training set) read the same way. Born in
002's step 7; its text is unchanged, so `PROMPT_VERSION` still names it.

An answer, one JSON line per article (written by `lab.llm.run`):

| field | |
|---|---|
| `id` | the article's Postgres id |
| `primary` | master slug, or null when none fits |
| `secondary` | 0 to 2 other master slugs (`clean` drops the primary and caps at 2) |
| `subcategories` | `master.subcategory` keys, 1 to 3, empty when none fits |
| `missing_subcategory` | pt-BR name of a subcategory the primary master lacks, or null |
| `model`, `prompt_version`, `taxonomy_version`, `created_at`, `usage` | metadata standard |

Answers go to `<prefix>_<prompt hash>.jsonl` (`lab.llm.output_path`): a changed taxonomy changes
the prompt, and so starts a new file.
"""

from pathlib import Path
from typing import Literal

from pydantic import create_model

from lab import llm
from lab.taxonomy_v2 import CATEGORIES, SLUGS, SUBCATEGORIES, TAXONOMY_VERSION

PROMPT_VERSION = "002-validate-v1"


def render_taxonomy() -> str:
    """Every master with its facets and subcategories, as the teacher reads them."""

    lines = []
    for master in CATEGORIES:
        lines.append(f"\n## {master.slug} ({master.name}): {master.definition}")
        for kind, facet in master.facets:
            lines.append(f"  {kind} facet ({facet.axis}):")
            lines.extend(
                f"  - {master.slug}.{sub.slug} ({sub.name}): {sub.definition}"
                for sub in facet.subcategories
            )
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You classify news articles for a Brazilian civic-debate app.

Taxonomy: master categories, each with subcategories in facets. A primary facet splits the master
along one axis; a cross-cutting facet holds aspects an article may or may not have.
{render_taxonomy()}

Return:
- `primary`: the master the article is mainly about; null if none fits.
- `secondary`: 0 to 2 other masters the article substantially covers (a passing mention is not
  enough). Never repeat the primary.
- `subcategories`: 1 to 3 subcategories, only of the primary and secondary masters: normally one
  from the primary master's primary facet, plus a cross-cutting one when the article is
  substantially about that aspect. Empty when the primary is null or when none fits.
- `missing_subcategory`: when the article belongs to its primary master but no subcategory of it
  fits, a short Brazilian Portuguese name for the one that would; otherwise null.
Judge by the content, not by the outlet. Articles may be in any language."""

Classification = create_model(
    "Classification",
    primary=(Literal[SLUGS] | None, ...),
    secondary=(list[Literal[SLUGS]], ...),
    subcategories=(list[Literal[tuple(SUBCATEGORIES)]], ...),
    missing_subcategory=(str | None, ...),
)

METADATA = {"prompt_version": PROMPT_VERSION, "taxonomy_version": TAXONOMY_VERSION}


def output_path(directory: Path, prefix: str) -> Path:

    return llm.output_path(directory, prefix, SYSTEM_PROMPT)


def clean(record: dict) -> dict:
    """The primary is never also a secondary, and there are at most 2 secondaries."""

    secondary = [s for s in record["secondary"] if s != record["primary"]][:2]
    return {**record, "secondary": secondary}


def read(path: Path) -> list[dict]:
    """The last successful answer per article, cleaned."""

    return [clean(r) for r in llm.read_jsonl(path)]
