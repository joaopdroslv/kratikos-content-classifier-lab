"""Step 6 — consolidate the subject tags of each master into faceted subcategories.

Subcategories are multi-label and organised in **facets**, each split along ONE axis:
- a **primary** facet every article of the master should fall in (sports → modality: futebol,
  tênis…; politics → institution: eleições, legislativo…);
- an optional **cross-cutting** facet of aspects that recur across the primary ones (sports →
  transferências, gestão…), which an article may or may not have.
An article gets every subcategory its tags map to, from either facet. Mixing two axes in one list
("futebol" next to "transferências") is what made the first run's boundaries overlap; facets keep
each list clean and still let an article be "futebol" + "transferências".

Two phases per master, repeated while a rule is broken (up to `REVISIONS` times):
A. **Propose** (`CONSOLIDATE_MODEL`): the facets and their subcategories (slug, name, definition),
   seeing the master's tags with counts, every master's definition and the subcategories already
   accepted for the other masters, so that two masters do not both claim a subject.
B. **Assign** (`ASSIGN_MODEL`, one call per distinct tag): the tag goes to one subcategory of
   either facet or to `unassigned`, through an enum, so it cannot invent one.

Shares are computed HERE, per article (an article has a subcategory when any of its tags maps to
it), and checked. Size is a hard rule only at the bottom (MIN_SHARE); a primary subcategory above
MAX_SHARE is a warning, since the natural unit of an axis can be large (football in sports).

Output: `data/002/subcategories.json`; every proposal and its check in `consolidation_log.jsonl`;
tag assignments in `assignments.jsonl` (resumable, keyed by a hash of the proposal).

    uv run python experiments/002-taxonomy-proposal/06_consolidate.py
"""

import asyncio
import hashlib
import json
import os
from collections import Counter
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, create_model

from lab import llm, metadata
from lab.config import DATA_DIR
from lab.taxonomy_v2 import BY_SLUG, CATEGORIES, SLUGS

DIR = DATA_DIR / "002"
PROPOSE_MODEL = os.getenv("CONSOLIDATE_MODEL", "gpt-4.1")
ASSIGN_MODEL = os.getenv("ASSIGN_MODEL", "gpt-4.1-mini")
PRIMARY_RANGE = (3, 10)
CROSS_RANGE = (2, 6)
MIN_SHARE = 0.04
MAX_SHARE = 0.40  # warning only
MIN_COVERAGE = 0.90  # of articles with a primary subcategory
REVISIONS = 2
NO_SPLIT = {"lotteries"}
UNASSIGNED = "unassigned"
CATCH_ALL = ("outros", "outras", "geral", "gerais", "diversos", "other", "misc")
# v1 was the flat, single-list consolidation of the first run (data/002/run1/)
PROPOSE_PROMPT_VERSION = "002-consolidate-v2"
ASSIGN_PROMPT_VERSION = "002-assign-v1"


class Subcategory(BaseModel):

    slug: str
    name: str
    definition: str


class Facet(BaseModel):

    axis: str
    subcategories: list[Subcategory]


class Proposal(BaseModel):

    primary: Facet
    cross_cutting: Facet | None


_MASTERS = "\n".join(f"- {c.slug} ({c.name}): {c.definition}" for c in CATEGORIES)

PROPOSE_PROMPT = f"""You design the subcategories of one master category for a Brazilian
civic-debate app. Readers pick the subcategories they want to follow; articles are then classified
into them, and one article can have several subcategories.

All master categories of the app:
{_MASTERS}

You receive one master, the subject tags of a sample of its articles with their counts (main tags
name the core subject; aspect tags a second facet), and the subcategories already defined for the
OTHER masters.

Design the master's subcategories as facets:
- `primary`: ONE axis (state it in `axis`, e.g. "sport modality", "state institution", "type of
  crime") and {PRIMARY_RANGE[0]} to {PRIMARY_RANGE[1]} subcategories along it, mutually exclusive,
  that together cover nearly every article of the master.
- `cross_cutting` (optional, null if nothing recurs): ONE different axis of aspects that recur
  ACROSS the primary subcategories (e.g. in sports: transfers, club finances, refereeing), with
  {CROSS_RANGE[0]} to {CROSS_RANGE[1]} subcategories. An article may have none of them.
Never mix two axes inside one facet: "futebol" and "transferências" cannot sit in the same list.

Rules for every subcategory:
- A SUBJECT, never a place: no country, region, city, club, person, company, competition or event
  in its name or definition. The reader's location is handled by a separate filter.
- A reader would plausibly choose to follow it. Not a fragment of a single story, not a vague
  umbrella: no "outros", "geral", "sociedade e cultura", "temas diversos".
- At least {MIN_SHARE:.0%} of the master's articles (use the counts); merge smaller ones into a
  natural grouping (e.g. "esportes de combate"). A large subcategory is fine when it is the
  natural unit of the axis (football in sports).
- It belongs to THIS master by the definitions above, and does not duplicate a subcategory of
  another master; where a neighbour is close, the definition says which side a case falls on.
- `slug`: snake_case English. `name`: short Brazilian Portuguese display name. `definition`: one
  or two English sentences: what belongs, and what does not where a neighbour is close."""

ASSIGN_PROMPT = """You file one subject tag of a news article into a subcategory of its master
category. Pick the subcategory whose definition fits the tag best, from either facet, or
`unassigned` if none does."""


def load_articles() -> dict[str, list[list[str]]]:
    """Per master, each article's list of normalised tags."""

    by_master: dict[str, list[list[str]]] = {slug: [] for slug in SLUGS}
    for r in llm.read_jsonl(DIR / "tags.jsonl"):
        if r["master"]:
            tags = [t.strip().lower() for t in (r["main_tag"], r["aspect_tag"]) if t]
            by_master[r["master"]].append(tags)
    return by_master


def _subcategories(proposal: Proposal) -> list[tuple[str, Subcategory]]:

    facets = [("primary", proposal.primary)]
    if proposal.cross_cutting:
        facets.append(("cross_cutting", proposal.cross_cutting))
    return [(kind, s) for kind, facet in facets for s in facet.subcategories]


def propose(client: OpenAI, messages: list[dict]) -> tuple[Proposal, str]:

    completion = client.chat.completions.parse(
        model=PROPOSE_MODEL,
        temperature=0,
        messages=messages,
        response_format=Proposal,
        max_tokens=6_000,
    )
    message = completion.choices[0].message
    return message.parsed, message.content


def assign(slug: str, proposal: Proposal, tags: list[str]) -> dict[str, str]:

    subcategories = _subcategories(proposal)
    options = [s.slug for _, s in subcategories] + [UNASSIGNED]
    schema = create_model("Assignment", subcategory=(Literal[tuple(options)], ...))
    master = BY_SLUG[slug]
    definitions = "\n".join(
        f"- {s.slug} ({s.name}; {kind}): {s.definition}" for kind, s in subcategories
    )
    system = (
        f"{ASSIGN_PROMPT}\n\nMaster: {master.name}: {master.definition}\n\n"
        f"Subcategories:\n{definitions}"
    )
    digest = hashlib.sha256(proposal.model_dump_json().encode()).hexdigest()[:10]
    key = f"{slug}|{digest}|"
    out = DIR / "assignments.jsonl"
    asyncio.run(
        llm.run(
            [(key + tag, f"Tag: {tag}") for tag in tags],
            out=out,
            system=system,
            schema=schema,
            model=ASSIGN_MODEL,
            extra={"prompt_version": ASSIGN_PROMPT_VERSION},
            desc=f"assign {slug}",
        )
    )
    return {
        r["id"][len(key) :]: r["subcategory"]
        for r in llm.read_jsonl(out)
        if r["id"].startswith(key)
    }


def check(proposal: Proposal, assignment: dict[str, str], articles: list[list[str]]):

    subcategories = _subcategories(proposal)
    primary = {s.slug for kind, s in subcategories if kind == "primary"}
    per_article = [{assignment.get(t, UNASSIGNED) for t in tags} for tags in articles]
    n = len(articles)
    held = Counter(label for labels in per_article for label in labels)
    shares = {s.slug: held[s.slug] / n for _, s in subcategories}
    coverage = sum(bool(labels & primary) for labels in per_article) / n
    stats = {
        "coverage": coverage,
        "labels_per_article": sum(len(labels - {UNASSIGNED}) for labels in per_article)
        / n,
        "several_primary": sum(len(labels & primary) > 1 for labels in per_article) / n,
    }

    problems, warnings = [], []
    counts = Counter(kind for kind, _ in subcategories)
    if not PRIMARY_RANGE[0] <= counts["primary"] <= PRIMARY_RANGE[1]:
        problems.append(f"primary facet has {counts['primary']} subcategories")
    if counts["cross_cutting"] and not (
        CROSS_RANGE[0] <= counts["cross_cutting"] <= CROSS_RANGE[1]
    ):
        problems.append(
            f"cross-cutting facet has {counts['cross_cutting']} subcategories"
        )
    if coverage < MIN_COVERAGE:
        problems.append(
            f"only {coverage:.0%} of the articles get a primary subcategory"
        )
    for kind, s in subcategories:
        words = f"{s.slug} {s.name}".lower().replace("_", " ").split()
        if any(word in words for word in CATCH_ALL):
            problems.append(f"'{s.slug}' is a catch-all")
        if shares[s.slug] < MIN_SHARE:
            problems.append(
                f"'{s.slug}' ({kind}) holds {shares[s.slug]:.0%} of the articles"
            )
        elif kind == "primary" and shares[s.slug] > MAX_SHARE:
            warnings.append(f"'{s.slug}' holds {shares[s.slug]:.0%} of the articles")
    return shares, stats, problems, warnings


def consolidate(
    client: OpenAI, slug: str, articles: list[list[str]], accepted: list[str], log
) -> dict:

    master = BY_SLUG[slug]
    main = Counter(tags[0] for tags in articles)
    aspect = Counter(tags[1] for tags in articles if len(tags) > 1)
    tags = list(dict.fromkeys([*main, *aspect]))
    neighbours = "\n".join(accepted) or "(none yet)"
    messages = [
        {"role": "system", "content": PROPOSE_PROMPT},
        {
            "role": "user",
            "content": f"Master: {master.slug} ({master.name}): {master.definition}\n\n"
            f"Subcategories already defined for other masters:\n{neighbours}\n\n"
            f"{len(articles)} articles.\n\nMain tags:\n"
            + "\n".join(f"- {t} ({c})" for t, c in main.most_common())
            + "\n\nAspect tags:\n"
            + "\n".join(f"- {t} ({c})" for t, c in aspect.most_common()),
        },
    ]
    best = None
    for round_ in range(REVISIONS + 1):
        proposal, raw = propose(client, messages)
        assignment = assign(slug, proposal, tags)
        shares, stats, problems, warnings = check(proposal, assignment, articles)
        # a revision can come out worse than the round it revised: keep the best one
        if best is None or len(problems) < len(best[4]):
            best = (proposal, assignment, shares, stats, problems, warnings)
        log.write(
            json.dumps(
                {
                    "master": slug,
                    "round": round_,
                    "proposal": proposal.model_dump(),
                    "shares": shares,
                    "stats": stats,
                    "problems": problems,
                    "warnings": warnings,
                    "model": PROPOSE_MODEL,
                    "prompt_version": PROPOSE_PROMPT_VERSION,
                    "created_at": metadata.now(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        log.flush()
        print(f"  {slug} round {round_}: {problems or 'ok'}")
        if not problems:
            break
        uncovered = Counter(
            t
            for article in articles
            if not any(
                assignment.get(t) in {s.slug for s in proposal.primary.subcategories}
                for t in article
            )
            for t in article
        )
        messages += [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": "Revise the proposal. Share of articles per subcategory:\n"
                + "\n".join(f"- {k}: {v:.0%}" for k, v in shares.items())
                + f"\nArticles with a primary subcategory: {stats['coverage']:.0%}"
                + "\nProblems:\n"
                + "\n".join(f"- {p}" for p in problems)
                + "\nTags of articles with no primary subcategory:\n"
                + "\n".join(f"- {t} ({c})" for t, c in uncovered.most_common(60)),
            },
        ]

    proposal, assignment, shares, stats, problems, warnings = best
    by_subcategory: dict[str, list[str]] = {}
    for tag in tags:
        by_subcategory.setdefault(assignment.get(tag, UNASSIGNED), []).append(tag)

    def facet(kind: str, f: Facet | None) -> dict | None:
        if f is None:
            return None
        return {
            "kind": kind,
            "axis": f.axis,
            "subcategories": [
                {
                    **s.model_dump(),
                    "share": shares[s.slug],
                    "tags": by_subcategory.get(s.slug, []),
                }
                for s in sorted(f.subcategories, key=lambda s: -shares[s.slug])
            ],
        }

    return {
        "master": slug,
        "articles": len(articles),
        "problems": problems,
        "warnings": warnings,
        **stats,
        "facets": [
            f
            for f in (
                facet("primary", proposal.primary),
                facet("cross_cutting", proposal.cross_cutting),
            )
            if f
        ],
        "unassigned_tags": by_subcategory.get(UNASSIGNED, []),
    }


def main() -> None:

    by_master = load_articles()
    client = OpenAI()
    out, accepted = [], []
    with (DIR / "consolidation_log.jsonl").open("w", encoding="utf-8") as log:
        for slug in SLUGS:
            if slug in NO_SPLIT:
                continue
            result = consolidate(client, slug, by_master[slug], accepted, log)
            out.append(result)
            for f in result["facets"]:
                accepted.extend(
                    f"- {slug} / {s['name']}: {s['definition']}"
                    for s in f["subcategories"]
                )

    (DIR / "subcategories.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for master in out:
        print(
            f"\n## {master['master']} ({master['articles']} articles;"
            f" coverage {master['coverage']:.0%};"
            f" {master['labels_per_article']:.1f} labels/article)"
        )
        for p in master["problems"]:
            print(f"  PROBLEM: {p}")
        for w in master["warnings"]:
            print(f"  warning: {w}")
        for f in master["facets"]:
            print(f"  [{f['kind']}: {f['axis']}]")
            for s in f["subcategories"]:
                print(f"    {s['share']:4.0%}  {s['name']}  [{s['slug']}]")


if __name__ == "__main__":
    main()
