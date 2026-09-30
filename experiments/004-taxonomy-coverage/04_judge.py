"""Step 4 — an LLM judges every cluster of step 3, and groups 002's missing-subcategory names.

Per cluster, the teacher reads the whole taxonomy and the cluster's subjects and says whether it
is:
- `existing`: described by 1 to 3 existing subcategories;
- `new_subcategory` of an existing master, or a `new_master`: a real gap, with a proposed pt-BR
  name and definition;
- `noise`: no common subject.
Since a cluster is judged by its majority, the teacher also names a recurring subject that a
MINORITY of its topics share and no subcategory covers (`uncovered_minority`).

Separately, one call groups the free-text `missing_subcategory` names of 002's validation into
themes, so a gap named in several ways is counted once.

Output (`data/004/`): `judge_<prompt hash>.jsonl`, `missing_<prompt hash>.jsonl` (a new file
whenever the taxonomy or the prompt changes), `gaps.md`.

    uv run python experiments/004-taxonomy-coverage/04_judge.py [--limit N]
"""

import argparse
import asyncio
import hashlib
import os
from collections import Counter
from typing import Literal

import pandas as pd
from pydantic import create_model

from lab import llm, pricing
from lab.config import DATA_DIR
from lab.taxonomy_v2 import CATEGORIES, SLUGS, SUBCATEGORIES, TAXONOMY_VERSION

DIR = DATA_DIR / "004"
VALIDATION = DATA_DIR / "002" / "validation_7cb234171d.jsonl"  # 002 step 7
MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1")
SUBJECTS = 25
PROMPT_VERSION = "004-judge-v1"
VERDICTS = ("existing", "new_subcategory", "new_master", "noise")
RULES = """Rules of this taxonomy: no geography and no named entities (people, clubs, parties,
countries, events) in a subcategory, since the reader's location is a separate filter; a
subcategory is a recurring subject, never a single story."""


def render_taxonomy() -> str:

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


def cluster_prompt() -> str:

    return f"""You review the taxonomy of a Brazilian civic-debate news app.

Taxonomy: master categories, each with subcategories in facets.
{render_taxonomy()}

{RULES}

You get a cluster of news topics, grouped by embedding similarity. Decide, by the majority of
its topics:
- `existing`: 1 to 3 existing subcategories describe them (list them). A cluster about a single
  story, place or person is `existing` when a subcategory covers its subject (one war →
  geopolitics.wars). A master without subcategories (lotteries) is `existing` with none listed.
- `new_subcategory`: most topics share a recurring subject that belongs to an existing master
  but that no subcategory covers. Give the master, a short Brazilian Portuguese name and a
  one-sentence English definition written like the ones above.
- `new_master`: the recurring subject belongs to no master. Give a name and definition.
- `noise`: the topics share no subject.
`uncovered_minority`: if a recurring subject shared by a minority of the topics (several of
them, not one) has no subcategory, its short Brazilian Portuguese name; otherwise null.
Do not propose what an existing subcategory already covers, even under another name.
`reason`: one sentence."""


def missing_prompt() -> str:

    return f"""You review the taxonomy of a Brazilian civic-debate news app.

Taxonomy: master categories, each with subcategories in facets.
{render_taxonomy()}

{RULES}

You get the names a classifier proposed when no subcategory fitted an article, each with its
master and how many articles proposed it. Group the names that mean the same subject into themes.
Per theme: a short Brazilian Portuguese name, the master, the member names exactly as given, and
a verdict: `existing` (an existing subcategory already covers it: list it), `new_subcategory`,
or `noise` (too specific or not a recurring subject)."""


def cluster_schema():

    return create_model(
        "ClusterVerdict",
        verdict=(Literal[VERDICTS], ...),
        existing=(list[Literal[tuple(SUBCATEGORIES)]], ...),
        master=(Literal[SLUGS] | None, ...),
        name=(str | None, ...),
        definition=(str | None, ...),
        uncovered_minority=(str | None, ...),
        reason=(str, ...),
    )


def missing_schema():

    theme = create_model(
        "Theme",
        name=(str, ...),
        master=(Literal[SLUGS] | None, ...),
        members=(list[str], ...),
        verdict=(Literal["existing", "new_subcategory", "noise"], ...),
        existing=(list[Literal[tuple(SUBCATEGORIES)]], ...),
    )
    return create_model("Themes", themes=(list[theme], ...))


def cluster_items(topics: pd.DataFrame, total: int) -> list[tuple[str, str]]:

    items = []
    for name, members in topics.groupby("cluster"):
        nearest = Counter()
        for key, n in zip(members["nearest"], members["members"]):
            nearest[key] += n
        subjects = members.sort_values(
            ["distance", "members"], ascending=[True, False]
        )["subject"].head(SUBJECTS)
        items.append(
            (
                str(name),
                f"Cluster of {len(members)} topics, {members['members'].sum() / total:.2%}"
                f" of all articles. Nearest subcategories: "
                + ", ".join(k for k, _ in nearest.most_common(3))
                + ".\n\nTopic subjects, most typical first:\n"
                + "\n".join(f"- {s}" for s in subjects),
            )
        )
    return items


def missing_item() -> tuple[str, str]:

    names = Counter(
        (r["primary"], r["missing_subcategory"].strip().lower())
        for r in llm.read_jsonl(VALIDATION)
        if r.get("missing_subcategory")
    )
    return (
        "missing_names",
        "\n".join(
            f"- {name} (master: {master}, {n})"
            for (master, name), n in names.most_common()
        ),
    )


def report(verdicts: list[dict], themes: list[dict], topics: pd.DataFrame) -> str:

    share = topics.groupby("cluster")["members"].sum() / topics["members"].sum()
    df = pd.DataFrame(verdicts)
    df["share"] = df["id"].astype(int).map(share)
    df = df.sort_values("share", ascending=False)

    out = [
        "# 004 — gaps in the taxonomy\n",
        f"{len(df)} clusters, {df['share'].sum():.1%} of all articles, judged by `{MODEL}`"
        f" (taxonomy {TAXONOMY_VERSION}).\n",
        "| verdict | clusters | share of all articles |\n|---|---|---|",
    ]
    for verdict in VERDICTS:
        part = df[df["verdict"] == verdict]
        out.append(f"| {verdict} | {len(part)} | {part['share'].sum():.2%} |")

    out.append("\n## Candidate gaps (new subcategory or master), largest first\n")
    out.append("| cluster | share | master | name | definition | reason |")
    out.append("|---|---|---|---|---|---|")
    gaps = df[df["verdict"].isin(("new_subcategory", "new_master"))]
    for r in gaps.itertuples():
        out.append(
            f"| {r.id} | {r.share:.2%} | {r.master or '(new)'} | {r.name} | {r.definition}"
            f" | {r.reason} |"
        )

    out.append("\n## Uncovered minorities inside clusters, largest cluster first\n")
    out.extend(
        f"- {r.id} ({r.share:.2%}, {r.verdict}): {r.uncovered_minority}"
        for r in df[df["uncovered_minority"].notna()].itertuples()
    )

    out.append(
        "\n## Subcategories no cluster was judged to hold (share split evenly)\n"
    )
    held = Counter()
    for r in df[df["verdict"] == "existing"].itertuples():
        for key in r.existing:
            held[key] += r.share / len(r.existing)
    out.append(", ".join(k for k in SUBCATEGORIES if not held[k]) or "(none)")

    out.append("\n## Noise\n")
    out.extend(
        f"- {r.id} ({r.share:.2%}): {r.reason}"
        for r in df[df["verdict"] == "noise"].itertuples()
    )

    out.append("\n## Themes of 002's missing-subcategory names\n")
    out.append("| theme | master | names | verdict |\n|---|---|---|---|")
    for t in sorted(themes, key=lambda t: -len(t["members"])):
        target = ", ".join(t["existing"]) if t["verdict"] == "existing" else ""
        out.append(
            f"| {t['name']} | {t['master']} | {len(t['members'])}"
            f" | {t['verdict']} {target} |"
        )
    return "\n".join(out)


async def main(limit: int | None) -> None:

    topics = pd.read_parquet(DIR / "topics_clustered.parquet")
    items = cluster_items(topics, int(topics["members"].sum()))
    if limit:
        items = items[:limit]
    extra = {"prompt_version": PROMPT_VERSION, "taxonomy_version": TAXONOMY_VERSION}

    runs = {}
    for kind, system, schema, todo in (
        ("judge", cluster_prompt(), cluster_schema(), items),
        ("missing", missing_prompt(), missing_schema(), [missing_item()]),
    ):
        digest = hashlib.sha256(system.encode()).hexdigest()[:10]
        out = DIR / f"{kind}_{digest}.jsonl"
        await llm.run(
            todo,
            out=out,
            system=system,
            schema=schema,
            model=MODEL,
            extra=extra,
            desc=kind,
        )
        runs[kind] = llm.read_jsonl(out)

    usage = [r["usage"] for records in runs.values() for r in records]
    text = report(runs["judge"], runs["missing"][0]["themes"], topics)
    text += f"\n\nCost of the answers on file: US${pricing.cost(MODEL, usage):.2f}."
    (DIR / "gaps.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    asyncio.run(main(parser.parse_args().limit))
