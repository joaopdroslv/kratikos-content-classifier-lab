"""Step 5 — tag each article with a v2 master category and 1 or 2 geography-free subject tags.

The k-means clusters of step 3 split by geography and by single stories, because that is what the
vectors see (the Premier League and the Brasileirão share no names). The subcategories are
therefore induced from text instead, in the TnT-LLM way (Wan et al., 2024): the LLM writes short,
GENERIC subject tags per article — no country, city, club, person or event — and step 6
consolidates them into subcategories.

Subcategories are multi-label (an article can be "futebol" and "transferências"), so an article
gets a `main_tag` (the sport, institution or core subject) and an optional `aspect_tag` (what about
it), mirroring the primary and cross-cutting facets of step 6. A free list of "1 to 3 tags" came
back with 3 almost every time, the third a detail of the story; a nullable field is honoured,
by gpt-4.1 (not by gpt-4.1-mini).

Input: the 3,075 articles of 001 plus the extra articles of step 4.
Output: `data/002/tags.jsonl`.

    uv run python experiments/002-taxonomy-proposal/05_tag.py [--limit N]
"""

import argparse
import asyncio
import os
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from lab import llm
from lab.config import DATA_DIR
from lab.taxonomy_v2 import CATEGORIES, SLUGS

DIR = DATA_DIR / "002"
OUT = DIR / "tags.jsonl"
# gpt-4.1-mini filled `aspect_tag` on 39 of 40 trial articles ("superação", "gols"); gpt-4.1 on 24.
MODEL = os.getenv("TAG_MODEL", "gpt-4.1")
PROMPT_VERSION = "002-tag-v5"
TAG_RULE = (
    "1 to 5 words, Brazilian Portuguese, generic: no place, person, club, company, brand, "
    "competition or event names"
)

Slug = Literal[SLUGS]


class Tagging(BaseModel):

    master: Slug | None
    missing_master: str | None
    main_tag: str = Field(description=TAG_RULE)
    aspect_tag: str | None = Field(description=TAG_RULE + "; null unless substantial")


_DEFINITIONS = "\n".join(f"- {c.slug} ({c.name}): {c.definition}" for c in CATEGORIES)

SYSTEM_PROMPT = f"""You organise news articles for a Brazilian civic-debate app, where readers follow
the subjects they care about.

Master categories:
{_DEFINITIONS}

For the article, return:
- `master`: the single master category the article is mainly about. If none fits, null, and set
  `missing_master` to a short Brazilian Portuguese name for the category that would fit;
  otherwise `missing_master` is null.
- `main_tag`: the subject of the article INSIDE its master, as a reader would name a subject they
  follow: the sport, the institution, or the core subject ("futebol", "congresso", "feminicídio",
  "inteligência artificial").
- `aspect_tag`: a SECOND, different facet, only when the article is substantially about it and a
  reader following that facet would want this article: "futebol" + "transferências de
  jogadores", "congresso" + "reforma tributária". Otherwise null — for most articles the
  main tag says it all. Never the same idea in other words, never a detail of the story ("gols",
  "casos raros", "superação").

Rules for both tags:
- Generic: NO country, state, city, region, club, team, person, party, company, brand, product,
  show, lottery, competition or event names (not "Copa do Mundo", "Libertadores", "Mega-Sena",
  "Oscar", "Premier League"). Geography is handled elsewhere.
  Good: "futebol", "tênis", "transferências de jogadores", "ataques aéreos em guerras",
  "feminicídio", "lançamento de smartphones", "eleições para governador", "supremo tribunal".
  Bad: "mercado do Botafogo", "futebol inglês", "guerra na Ucrânia", "caso Isabelle", "iPhone 18".
- The level of a subject someone would follow for months, not of the single story: "crise
  financeira de clubes", not "dívida de um clube específico".
- A sport's name alone ("futebol", "automobilismo") is a good tag: it is a subject people follow.
- Judge by the content, not by the outlet. Articles may be in any language."""


def load_articles() -> pd.DataFrame:

    base = pd.read_parquet(DATA_DIR / "001" / "sample.parquet")
    extra = pd.read_parquet(DIR / "extra_sample.parquet")
    return pd.concat([base, extra], ignore_index=True).drop_duplicates("id")


async def main(limit: int | None) -> None:

    articles = load_articles()
    if limit:
        articles = articles.head(limit)
    items = [
        (row["id"], llm.render_article(row)) for row in articles.to_dict("records")
    ]
    await llm.run(
        items,
        out=OUT,
        system=SYSTEM_PROMPT,
        schema=Tagging,
        model=MODEL,
        extra={"prompt_version": PROMPT_VERSION},
        desc="tagging",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    asyncio.run(main(parser.parse_args().limit))
