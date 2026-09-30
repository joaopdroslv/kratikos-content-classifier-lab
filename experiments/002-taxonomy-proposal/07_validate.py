"""Step 7 — validate the proposed taxonomy on articles no earlier step has seen.

A fresh uniform sample (the unbiased view of the corpus) plus a few articles per small master is
classified by the teacher with the WHOLE curated v2 taxonomy (`lab.taxonomy_v2`): a primary master, up to two secondary ones,
and 1 to 3 subcategories from any facet of those masters, with "none fits" allowed at every level.
Prompt, schema and answer format are `lab.classify_v2`, which was extracted from this step.
This measures, per master:
- coverage: articles that get a primary-facet subcategory of their primary master;
- the share of every subcategory (tiny ones are candidates to merge or drop);
- labels per article, and which subcategories of the SAME facet co-occur (a blurry boundary);
- subcategories chosen outside the article's masters (a blurry boundary between masters).
Also: the v2 master distribution on the uniform stratum.

Output: `data/002/validation_sample.parquet`, `data/002/validation_<prompt hash>.jsonl` (a new
file whenever the taxonomy changes, so answers to an old one never mix in), `validation.md`.

    uv run python experiments/002-taxonomy-proposal/07_validate.py [--limit N]
"""

import argparse
import asyncio
import os
from collections import Counter
from itertools import combinations

import pandas as pd
from sqlalchemy import text

from lab import classify_v2, llm, metadata
from lab.config import DATA_DIR, get_engine
from lab.taxonomy_v2 import SLUGS, SUBCATEGORIES

DIR = DATA_DIR / "002"
MODEL = os.getenv("VALIDATE_MODEL", "gpt-4.1")
UNIFORM_SIZE = 1_200
PER_SMALL_MASTER = 40
SMALL_V1 = ("housing", "education", "transport", "human_rights", "culture")
CONTENT_CHARS = 1_500
SEED = 0.17


def load_taxonomy() -> dict[str, dict]:
    """`master.subcategory` -> {master, kind, name}, from the curated `lab.taxonomy_v2`."""

    return {
        key: {"master": master.slug, "kind": kind, "name": sub.name}
        for key, (master, kind, sub) in SUBCATEGORIES.items()
    }


def draw_sample() -> pd.DataFrame:

    path = DIR / "validation_sample.parquet"
    if path.exists():
        return pd.read_parquet(path)
    used = set(pd.read_parquet(DATA_DIR / "001" / "sample.parquet")["id"]) | set(
        pd.read_parquet(DIR / "extra_sample.parquet")["id"]
    )
    topics = pd.read_parquet(DIR / "topics_categorised.parquet")
    select = f"""
        SELECT
            na.id::text AS id,
            na.title,
            na.description,
            LEFT(na.content, {CONTENT_CHARS}) AS content,
            na.category AS source_category
        FROM public.news_articles AS na
        JOIN ai.ingested_news_articles AS ina ON TRUE
            AND ina.content_id = na.id
            AND ina.successful IS TRUE
    """
    parts = []
    with get_engine().connect() as conn:
        conn.execute(text("SELECT setseed(:seed)"), {"seed": SEED})
        uniform = pd.read_sql(
            text(
                f"{select} WHERE NOT (na.id::text = ANY(:used)) ORDER BY random() LIMIT :n"
            ),
            conn,
            params={"used": list(used), "n": UNIFORM_SIZE},
        )
        parts.append(uniform.assign(stratum="uniform"))
        used |= set(uniform["id"])
        for category in SMALL_V1:
            topic_ids = topics.loc[topics["category"] == category, "id"].tolist()
            part = pd.read_sql(
                text(f"""
                    {select}
                    JOIN ai.topic_members AS tm ON TRUE
                        AND tm.content_id = na.id
                        AND tm.content_type = 'news_article'
                    WHERE TRUE
                        AND tm.topic_id = ANY(CAST(:topics AS uuid[]))
                        AND NOT (na.id::text = ANY(:used))
                    ORDER BY random()
                    LIMIT :n
                    """),
                conn,
                params={"topics": topic_ids, "used": list(used), "n": PER_SMALL_MASTER},
            )
            parts.append(part.assign(stratum=f"extra:{category}"))
            used |= set(part["id"])
    sample = pd.concat(parts, ignore_index=True).drop_duplicates("id")
    sample["sampled_at"] = metadata.now()
    sample.to_parquet(path, index=False)
    return sample


def summarise(records: list[dict], sample: pd.DataFrame, index: dict[str, dict]) -> str:

    df = sample.merge(pd.DataFrame(records), on="id")
    uniform = df[df["stratum"] == "uniform"]
    out = ["# 002 — validation of the proposed taxonomy\n"]

    out.append(f"{len(df)} articles ({len(uniform)} uniform).\n")
    out.append("## Primary master, uniform stratum\n")
    shares = uniform["primary"].fillna("(none)").value_counts(normalize=True)
    out.extend(f"- {k}: {v:.1%}" for k, v in shares.items())

    out.append("\n## Per master (all strata, by primary master)\n")
    out.append(
        "| master | n | coverage | none fits | labels/article | outside masters |\n|---|---|---|---|---|---|"
    )
    rows = []
    for master in SLUGS:
        part = df[df["primary"] == master]
        if part.empty:
            continue
        primary_subs = {
            k
            for k, v in index.items()
            if v["master"] == master and v["kind"] == "primary"
        }
        subs = part["subcategories"].map(set)
        allowed = part.apply(lambda r: {r["primary"], *r["secondary"]}, axis=1)
        outside = [
            any(index.get(s, {}).get("master") not in a for s in ss)
            for ss, a in zip(subs, allowed)
        ]
        rows.append(
            f"| {master} | {len(part)} | {subs.map(lambda s: bool(s & primary_subs)).mean():.0%}"
            f" | {part['missing_subcategory'].notna().mean():.0%}"
            f" | {subs.map(len).mean():.1f} | {sum(outside) / len(part):.0%} |"
        )
    out.extend(rows)

    out.append("\n## Subcategory shares (articles of the master that have it)\n")
    for master in SLUGS:
        part = df[df["primary"] == master]
        if part.empty:
            continue
        held = Counter(s for ss in part["subcategories"] for s in ss)
        out.append(f"\n### {master} ({len(part)})")
        for key, v in index.items():
            if v["master"] == master:
                out.append(f"- {held[key] / len(part):4.0%}  {v['name']} ({v['kind']})")
        missing = Counter(m.lower() for m in part["missing_subcategory"].dropna())
        if missing:
            out.append(
                "- missing: "
                + ", ".join(f"{k} ({n})" for k, n in missing.most_common(8))
            )

    out.append("\n## Same-facet co-occurrence (blurry boundaries), top 20\n")
    pairs = Counter()
    for ss in df["subcategories"]:
        for a, b in combinations(sorted(set(ss)), 2):
            ia, ib = index.get(a), index.get(b)
            if ia and ib and ia["master"] == ib["master"] and ia["kind"] == ib["kind"]:
                pairs[(a, b)] += 1
    out.extend(f"- {n:3d}  {a} + {b}" for (a, b), n in pairs.most_common(20))
    return "\n".join(out)


async def main(limit: int | None) -> None:

    index = load_taxonomy()
    sample = draw_sample()
    if limit:
        sample = sample.head(limit)
    out = classify_v2.output_path(DIR, "validation")
    await llm.run(
        [(r["id"], llm.render_article(r)) for r in sample.to_dict("records")],
        out=out,
        system=classify_v2.SYSTEM_PROMPT,
        schema=classify_v2.Classification,
        model=MODEL,
        extra=classify_v2.METADATA,
        desc="validate",
    )
    ids = set(sample["id"])
    records = [r for r in classify_v2.read(out) if r["id"] in ids]
    report = summarise(records, sample, index)
    (DIR / "validation.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    asyncio.run(main(parser.parse_args().limit))
